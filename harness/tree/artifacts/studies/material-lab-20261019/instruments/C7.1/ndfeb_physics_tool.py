"""NdFeB SPS physics-model tool: predict, p_meets_spec, eig, condition.

Evidence class: owner-built conditional executable physics model (Dynamical's model fitted to RPI's released records
and declared literature priors; not an RPI or collaborator model); not a physical observation; does not calibrate
current-body coercivity. The fit's assumptions about those records are listed in the README ("Fitted assumptions").

Dependencies: Python standard library + numpy. No JAX. Loads hash-bound artifacts:
  physics.py   - schedule discretisation and rate laws (identical bytes to the fitted model)
  forward.py   - mechanism forward equations (the fitted model's, plus the C4 feedstock particle-size layer and the C5
                 SEM/BSE observables and milled-feedstock density layer: declared, uncalibrated hooks the fit never
                 reached)
  posterior    - .npz of thinned posterior and prior draws + metadata
  microstructure.py - C6 microstructure layer: declared link and measurement functions of the SEM grain and feature
                 size, SEM porosity and Nd-rich phase outputs, and the descriptive SPS trace (pure numpy)
  coercivity_link.py - optional (C6 work package 1): the coercivity link; without it the link outputs are omitted
  manifest     - TOOL_MANIFEST.json with the sha256 of this module and the files above (verified on load)

C6 (physics tool C6, microstructure layer): every C5 output is computed exactly as in C5 for the same seed; the new
outputs (grain_sem_dist, fracture_intercept_um, porosity_area_fraction_sem, the Nd-rich phase outputs, sps_log and the
coercivity link) use their own generators and the per-draw parameters c6_* stored next to the C5 coercivity draws.

C7 (material lab, 2026-10-23; README.md in this package lists every change and its evidence):
  - density: the C5.0 SPS density draws reweighted on #7 (7.01, Archimedes) and #8 as sintered (7.07, method not
    stated), one shared 850 C batch PREP-C850 (posterior_tool.npz; build/c7_density_update.py)
  - anneal: a load-free anneal can densify a body (anneal.py, anneal_posterior.npz; four mechanisms calibrated on #8's
    one paired return, build/c7_anneal_fit.py). Every annealed stage carries its own density: route-level density_rel,
    C5 remanence (relative to historical E's own anneal), the coercivity link's Br, the fracture-intercept feature
    factor. The 900 C hold is a route key (anneal_hold_h, default 2 h = the planned program, or "as_#8" for the same
    unconfirmed program as #8); the C5 grain law's 900 C growth follows the hold
  - inputs: unknown route keys raise with suggestions; degassing_completeness is an echoed alias of degas_completeness;
    alignment_cos and alignment_cos_theta are one physical <cos theta> feeding both magnetic paths (a conflict raises);
    predict() reports route_resolution
  - joint_target(): both magnetic paths (C5 and the coercivity link) on one set of draws, with the boundary-film
    magnetization and every epistemic quantity shared by the bodies of one preparation, body scatter from C5, the same
    declared temperature scenario, and the structural Br^2/4 bound reported
  - everything not named above is computed exactly as in C6 for the same seed, posterior and route

C7.1 (2026-10-23, corrective; README.md "C7.1"):
  - anneal.py: CAPACITY's removable porosity is a state of the whole executed sequence (segmenting one program no longer
    removes more); CLOSED is integrated on a fixed largest dose step (segmentation-independent); the tool evaluates every
    stage from the as-sintered body through its whole sequence
  - one physical dependence on the 900 C hold on both magnetic paths: the hold-scaled C5 grain now also enters C5's
    grain-size mechanism H4 (Kronmueller term), as it enters the link's N_eff; every non-grain magnetic anneal term
    (C5's fitted factors, the link's re-wetting and 520 C gain) is historical E's 2 h response on both paths (declared)
  - polished-section planarity and SEM porosity read the body's density at the route's stage, as in the twin
  - eig() refuses what its latents cannot represent (annealed-stage density, Br and grain; annealed goals);
    anneal_eig() and condition_anneal() select and update anneal experiments on the same layer, readings and
    nuisances as predict() and the twin

Usage inside a science workspace (files are mounted by sha256, without names or extensions):
    import importlib.util
    from importlib.machinery import SourceFileLoader
    raw = '/workspace/raw/'
    loader = SourceFileLoader('ndfeb_physics_tool', raw + TOOL_SHA)
    spec = importlib.util.spec_from_loader('ndfeb_physics_tool', loader)
    tool = importlib.util.module_from_spec(spec); loader.exec_module(tool)
    pm = tool.load(physics=raw + PHYSICS_SHA, forward=raw + FORWARD_SHA, posterior=raw + POSTERIOR_SHA,
                   manifest=raw + MANIFEST_SHA, microstructure=raw + MICROSTRUCTURE_SHA)
    pm.predict({'T_peak_C': 800, 'pressure_mode': 'cooling', 'P_MPa': 60})

Units: deg C, minutes, MPa, relative density (fraction of 7.62 g/cm3), kOe, kGs, MGOe, um.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

TOOL_VERSION = "ndfeb-sps-physics-tool-20261023.8.1"
EVIDENCE_CLASS = ("owner-built conditional executable physics model (Dynamical's model fitted to RPI's released records "
                  "and declared literature priors; not an RPI or collaborator model); not a physical observation; does "
                  "not calibrate current-body coercivity")
OBJECTIVE = {"Hci_kOe_min": 20.0, "BHmax_MGOe_min": 45.0, "note": "Dy/Tb-free, verified at operating temperature; "
             "10 kOe / 11 MGOe is a historical simulated benchmark from sample E, not the objective"}
OBJECTIVE_SPEC = {"Hci_kOe_min": 20.0, "BHmax_MGOe_min": 45.0}   # C7 joint_target default (room temperature)
RHO_TD = 7.62
HERE = Path(__file__).resolve().parent
_PKG = "ndfeb_physics_model_c71"  # distinct from C6's and C7's synthetic packages, so every version loads in one process

QUANTILE_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
QUANTILE_KEYS = ("q01", "q025", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q975", "q99")
H_MEASUREMENT_LOG_SD = 0.25  # declared, uncalibrated: hydrogen content vs the model's residual-hydrogen index
H_DETECTION_INDEX = 0.01  # declared detection limit of a hydrogen measurement on the residual-hydrogen index scale
# (1 = non-degassed powder); a body below it (e.g. degas_completeness "complete") reads at the limit
# degas_completeness d is a declared, uncalibrated input (a number, a uniform or beta distribution, or "complete").
# The route defaults below are the v4 build defaults; a package may override them through meta["route_defaults"]
# (each time slice states its own defaults from information released by its cutoff).
ROUTE_DEFAULTS = dict(T_peak_C=None, heating_rate_C_per_min=10.0, dwell_min=5.0, P_MPa=60.0, pressure_mode="cooling",
                      cooling_rate_C_per_min=100.0, load_ramp_min=0.3, degas=1.0, anneal="none", alignment_cos=0.5,
                      T_measure_C=20.0, prep="new", degas_completeness=1.0, current_path="compact",
                      feedstock_d50_um=4.0)
# C4 feedstock particle-size layer (declared, uncalibrated): feedstock_d50_um = 4.0 is RPI's stated feedstock, the
# feedstock of every fitted record; any other value is a declared extrapolation (forward.py, "C4 feedstock").
FEEDSTOCK_KEY = "feedstock_d50_um"
FEEDSTOCK_REF_UM = 4.0
FEEDSTOCK_RANGE_UM = (0.5, 8.0)
FEEDSTOCK_PARAMS = ("fd_beta", "fd_sg", "fd_dox", "fd_gox", "fd_kH")
STATUS_FITTED = "fitted_feedstock"
STATUS_DECLARED = "declared_extrapolation_uncalibrated"
OUTPUTS = ("density_rel", "grain_um", "Hci_kOe", "Br_kGs", "Hk_over_Hci", "BHmax_MGOe")
# C5 (declared, uncalibrated): SEM/BSE outputs of predict(), observables of condition(), and the milled-feedstock density
# layer. Per-draw parameters c5_* are stored prior draws (forward.py, "C5 SEM/BSE observables"); the package's status
# strings for them come from its posterior metadata (meta["c5"]["status"]).
SEM_STAGES = ("as_sintered", "after_900_520")  # both stages of one piece (SEM grain size and polished-section BSE)
STAGE_ALIASES = {"900+520": "after_900_520"}  # accepted on input
SEM_OUTPUTS = ("grain_size_sem_um", "anneal_grain_change", "gb_phase_bse")
BSE_CLASSES = ("spread", "pooled", "unreadable")
C5_SEM_PARAMS = ("c5_cm", "c5_zeta", "c5_kox", "c5_Wstar", "c5_tau", "c5_rann")
C5_DENSITY_PARAMS = ("c5_nD", "c5_gam", "c5_cox", "c5_dox", "c5_zfd")
STATUS_MODELLED_PAIR = "modelled_pair"
GRAIN_FIELD_LOG_SD = 0.15  # declared: field sampling of a reported mean grain size (log-sd), as forward.py
E_RHO_PLACEHOLDER = 0.9738  # density passed for SEM/BSE-only observations in condition(); grain and W ignore it
# C6 (microstructure layer; declared link functions in microstructure.py, per-draw parameters c6_* stored next to the
# C5 coercivity draws, statuses from the package's posterior metadata meta["c6"]["status"]). New outputs draw from
# their own generators (seed, 61 / 67), so every C5 array of a call is unchanged.
C6_STAGES = ("as_sintered", "after_900", "after_900_520")
C6_OUTPUTS = ("grain_sem_dist", "fracture_intercept_um", "porosity_area_fraction_sem", "nd_rich_area_fraction",
              "pocket_ecd_um", "pocket_share", "elongated_share", "faint_line_coverage", "sps_log", "coercivity_link")
ALIGN_KEY = "alignment_cos_theta"  # A6: grain alignment <cos theta> for the coercivity link only (C5 ignores it)
ALIGN_DEFAULT = 0.5                # isotropic: every record to date
ALIGN_RANGE = (0.5, 0.95)
J_GB_POSTERIOR_T = {"0.0": 0.26, "0.6": 0.65, "1.2": 0.08}  # Phase 2a D/E posterior of the film magnetization (never drawn)
LINK_STATUS_MISSING = "link module not installed"
LINK_STATE_KEYS = ("n", "route", "alignment_cos_theta", "alignment_declared", "stages", "rho_rel", "grain_model_um",
                   "grain_sigma_ln", "grain_sem_median_um", "fracture_intercept_um", "nd_rich", "W", "c5",
                   "J_GB_posterior_T", "ln_hci_chem_offset", "rho_rel_by_stage")
# requests no C6 instrument answers (for the campaign twin, C6_SPEC C): key -> reason
REFUSALS = {"powder_icp_oes": "no model of powder chemistry by ICP-OES",
            "film_magnetization": "the 2-3 nm boundary film's magnetization is not observable by any measurement in "
                                  "this window (STEM-EDS or atom-probe tomography can constrain chemistry, not directly measure film magnetization)",
            "stem_eds": "no STEM or atom-probe instrument is simulated in this window"}
PRESSURE_MODES = ("heating", "cooling", "dwell")
ANNEALS = ("none", "900", "900+520")
# C7 route input handling: every key must be a route key, an accepted alias (applied and echoed) or informational
ANNEAL_HOLD_KEY = "anneal_hold_h"     # hold of the 900 C step (h); "as_#8" = the same unconfirmed program as #8
ANNEAL_HOLD_DEFAULT = 2.0             # the September timeline's planned 900 C / 2 h (not confirmed for #8)
ANNEAL_HOLD_RANGE = (0.5, 4.0)
ANNEAL_HOLD_AS8 = "as_#8"
SECOND_STEP = (520.0, 2.0)            # the planned 520 C / 2 h step (below the Nd-rich eutectic: no densification)
ROUTE_ALIASES = {"degassing_completeness": "degas_completeness"}   # the campaign twin's preparation key
INFORMATIONAL_KEYS = ("label", "body_id", "note")                  # echoed, never read
E_ROUTE_ANNEAL_RHO = 0.9738           # historical E's as-sintered relative density (7.42): C5's anneal factors are E's
C7_STAGE_DENSITY_NOTE = ("C7: a load-free anneal can densify a body (anneal.py). density_rel is the body's density at "
                         "the route's stage; density_rel_as_sintered is the same body before the anneal. Br at an "
                         "annealed stage is C5's (whose anneal factor was fitted on historical E) times the body's "
                         "density gain over E's gain under the same anneal draw; ln Hci keeps C5's as-sintered "
                         "porosity (declared: porosity damage is set at sintering)")
CURRENT_PATHS = ("compact", "insulated")
LT_LEVELS = ("trap", "load", "current")
SPEC_KEYS = ("Hci_kOe_min", "BHmax_MGOe_min", "Br_kGs_min", "density_rel_min", "T_operating_C")


def _validate_spec(spec):
    """Reject unknown or non-numeric spec keys (a misspelled key must not silently drop a requirement)."""
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"spec must be a non-empty dict with keys from {SPEC_KEYS}")
    bad = sorted(k for k in spec if k not in SPEC_KEYS)
    if bad:
        raise ValueError(f"unknown spec keys {bad}; allowed: {list(SPEC_KEYS)}")
    for k, v in spec.items():
        if not _is_number(v):
            raise ValueError(f"spec[{k!r}] must be a number")
    return spec


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_module(name, path):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _compose(physics_path, forward_path):
    """Compose physics.py and forward.py into a synthetic package (forward does `from . import physics`)."""
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = []
    sys.modules[_PKG] = pkg
    ph = _load_module(_PKG + ".physics", physics_path)
    pkg.physics = ph
    fw = _load_module(_PKG + ".forward", forward_path)
    return ph, fw


def load(physics=None, forward=None, posterior=None, manifest=None, verify=True, microstructure=None,
         coercivity_link=None, anneal=None, anneal_posterior=None):
    """Load the tool. With a manifest (default: TOOL_MANIFEST.json next to this file) every artifact's sha256 is
    checked and a mismatch raises. Without one the model loads with `verified = False`, reported in every output.
    C6: microstructure (default microstructure.py next to this file) carries the C6 link functions; coercivity_link
    (default coercivity_link.py next to this file, when it exists) is the coercivity link of work package 1. A manifest
    that lists either file binds it too; without the link module the link outputs are omitted with status
    "link module not installed".
    C7: anneal (default anneal.py) and anneal_posterior (default anneal_posterior.npz) carry the anneal layer; both are
    required together, and a manifest that lists them binds them. Without them annealed stages keep the as-sintered
    density, as in C6, and every output says so."""
    physics = physics or HERE / "physics.py"
    forward = forward or HERE / "forward.py"
    posterior = posterior or HERE / "posterior_tool.npz"
    if microstructure is None and (HERE / "microstructure.py").exists():
        microstructure = HERE / "microstructure.py"
    if coercivity_link is False:   # explicit opt-out: load without the link even when coercivity_link.py exists
        coercivity_link = None
    elif coercivity_link is None and (HERE / "coercivity_link.py").exists():
        coercivity_link = HERE / "coercivity_link.py"
    if anneal is False:            # explicit opt-out of the C7 anneal layer (C6 behaviour for annealed stages)
        anneal, anneal_posterior = None, None
    else:
        if anneal is None and (HERE / "anneal.py").exists():
            anneal = HERE / "anneal.py"
        if anneal_posterior is None and anneal is not None and (HERE / "anneal_posterior.npz").exists():
            anneal_posterior = HERE / "anneal_posterior.npz"
    if (anneal is None) != (anneal_posterior is None):
        raise ValueError("the C7 anneal layer needs both anneal.py and anneal_posterior.npz")
    if manifest is None and (HERE / "TOOL_MANIFEST.json").exists():
        manifest = HERE / "TOOL_MANIFEST.json"
    hashes = {"physics": _sha256(physics), "forward": _sha256(forward), "posterior": _sha256(posterior),
              "tool": _sha256(Path(__file__))}
    if microstructure is not None:
        hashes["microstructure"] = _sha256(microstructure)
    if coercivity_link is not None:
        hashes["coercivity_link"] = _sha256(coercivity_link)
    if anneal is not None:
        hashes["anneal"] = _sha256(anneal)
        hashes["anneal_posterior"] = _sha256(anneal_posterior)
    verified = False
    if verify and manifest is not None:
        man = json.loads(Path(manifest).read_text())
        missing = [k for k in ("physics", "forward", "posterior", "tool") if k not in man.get("files", {})]
        if missing:
            raise ValueError(f"manifest lacks entries {missing}; a v3.1 manifest lists physics, forward, posterior and tool")
        if man.get("tool_version") != TOOL_VERSION:
            raise ValueError(f"manifest is for {man.get('tool_version')}, this module is {TOOL_VERSION}")
        opt = ("microstructure", "coercivity_link", "anneal", "anneal_posterior")
        absent = [k for k in opt if k in man["files"] and k not in hashes]
        if absent:
            raise ValueError(f"the manifest lists {absent}: pass the file(s) to load()")
        unbound = [k for k in opt if k in hashes and k not in man["files"]]
        if unbound:
            raise ValueError(f"{unbound} loaded but not listed in the manifest (a C7 manifest binds every module)")
        keys = ("physics", "forward", "posterior", "tool") + tuple(k for k in opt if k in hashes)
        want = {k: man["files"][k]["sha256"] for k in keys}
        bad = {k: (hashes[k], want[k]) for k in want if hashes[k] != want[k]}
        if bad:
            raise ValueError(f"artifact hashes differ from the manifest: {bad}")
        verified = True
    ph, fw = _compose(physics, forward)
    ms = _load_module(_PKG + ".microstructure", microstructure) if microstructure is not None else None
    link = _load_module(_PKG + ".coercivity_link", coercivity_link) if coercivity_link is not None else None
    an = _load_module(_PKG + ".anneal", anneal) if anneal is not None else None
    return PhysicsModel(ph, fw, posterior, hashes, verified=verified, ms=ms, link=link, an=an,
                        anneal_posterior=anneal_posterior)


def _n_logpdf(y, mu, s):
    return -0.5 * ((y - mu) / s) ** 2 - np.log(s) - 0.5 * math.log(2 * math.pi)


def _t_logpdf(y, mu, s, nu=5.0):
    z = (y - mu) / s
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - np.log(s)
            - (nu + 1) / 2 * np.log1p(z ** 2 / nu))


def _lse(a, axis=None):
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(np.squeeze(out))


def _lme(a, axis):
    a = np.asarray(a, float)
    return _lse(a, axis) - math.log(a.shape[axis])


def _iqr(x):
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return float(q75 - q25)


def _distribution(x):
    """Quantiles (the primary summary) and a quantile-matched log-normal approximation of a predictive sample
    (mu_ln = ln median, sd_ln = half the 15.9-84.1 % log spread, over the positive draws), with its adequacy:
    lognormal_max_log_error is the largest |ln q - (mu_ln + z_q sd_ln)| over q05..q95, and lognormal_adequate is true
    when it is below 0.25. Mixtures over mechanisms and degassing states are often not log-normal."""
    from statistics import NormalDist
    x = np.asarray(x, float)
    qs = np.quantile(x, QUANTILE_LEVELS)
    pos = x[x > 0]
    mu = sd = worst = None  # JSON-safe: None where a log-normal summary is not computable (never NaN or inf)
    if len(pos) >= 10:  # quantile-matched (robust): a few collapsed draws must not distort the summary
        l16, l50, l84 = np.log(np.quantile(pos, [0.158655, 0.5, 0.841345]))
        mu, sd = float(l50), float((l84 - l16) / 2.0)
        central = [(q, lv) for q, lv in zip(qs, QUANTILE_LEVELS) if 0.05 <= lv <= 0.95]
        if sd > 0 and all(q > 0 for q, _ in central):
            worst = float(max(abs(math.log(q) - (mu + NormalDist().inv_cdf(lv) * sd)) for q, lv in central))
    return dict(quantiles={k: float(v) for k, v in zip(QUANTILE_KEYS, qs)}, mu_ln=mu, sd_ln=sd,
                lognormal_max_log_error=worst, lognormal_adequate=bool(worst is not None and worst < 0.25),
                frac_nonpositive=float(np.mean(x <= 0)))


def _load_on_cooling(r):
    """Hull feature: 1 load on cooling, 0 load in heating, 0.5 load from the start of the dwell (declared scenario)."""
    return {"cooling": 1.0, "heating": 0.0, "dwell": 0.5}[r["pressure_mode"]]


def _is_number(v):
    import numbers
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _check_prep_feedstock(seen, r, what="routes"):
    """Routes (or observations) sharing a preparation share one powder batch, hence one feedstock size."""
    v = float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM))
    if seen.setdefault(r["prep"], v) != v:
        raise ValueError(f"{what} sharing preparation {r['prep']!r} must share {FEEDSTOCK_KEY} (one powder batch)")


def _scenarios(r):
    dc = r["degas_completeness"]
    dc_num = float(dc) if _is_number(dc) else None
    complete = dc == "complete"
    extrap = complete or (dc_num is not None and dc_num > 1.0) or (isinstance(dc, dict) and (
        (dc.get("distribution") == "uniform" and float(dc.get("high", 0)) > 1.0)))
    calibrated = bool(abs(float(r["alignment_cos"]) - 0.5) < 1e-9 and abs(float(r["T_measure_C"]) - 20) < 1e-9
                      and dc_num is not None and dc_num in (0.0, 1.0) and r["current_path"] == "compact"
                      and r["pressure_mode"] in ("heating", "cooling")
                      and float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM)) == FEEDSTOCK_REF_UM)
    return {"alignment_cos": float(r["alignment_cos"]), "T_measure_C": float(r["T_measure_C"]),
            "degas_occurrence_probability": float(r["degas"]), "degas_completeness": dc,
            "degas_extrapolation_beyond_fitted_state": bool(extrap),
            "degas_complete_declared": bool(complete), "current_path": r["current_path"],
            "pressure_mode_dwell_declared": r["pressure_mode"] == "dwell",
            FEEDSTOCK_KEY: float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM)),
            "feedstock_declared": float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM)) != FEEDSTOCK_REF_UM,
            "feedstock_note": ("feedstock_d50_um other than 4 um (the feedstock of every fitted record) is a declared, "
                               "uncalibrated extrapolation: grain size, coercivity (an empirical grain-size factor in "
                               "every mechanism, an oxidation term and a hydrogen term), density (a declared size, "
                               "packing and oxide layer on the densification rates and green density), the SEM/BSE "
                               "outputs and the flags change"),
            "calibrated": calibrated,
            "note": "model assumptions, not records: calibrated only at isotropic alignment, room temperature, the "
                    "two degassing states the fit assigns (the model assumes A-C correspond to completeness 0 and D/E "
                    "to completeness 1; RPI's record does not state A-C's dehydrogenation), a current-carrying compact "
                    "(conductive graphite die) and load in heating or on cooling. Alignment, temperature, intermediate "
                    "completeness (interpolated), completeness above 1 (extrapolated), complete degassing "
                    "(no residual hydrogen; generic degassing terms held at the fitted D/E state), an insulated "
                    "compact and load from the start of the dwell are declared scenarios; batch records or a hydrogen "
                    "measurement would pin the completeness."}


# ---------------------------------------------------------------------------------------------------- eig (v4) helpers
EIG_METHOD_V4 = "bracketed-ba-v4"
EIG_QUANTITIES = ("Hci", "Br", "sq", "density", "grain", "H_ppm", "Hci_ratio_T")
EIG_EPS = 0.01            # every fitted posterior is mixed with the prior weights at this rate (bounds log q below)
EIG_ESS_POOL = (20.0, 200.0)  # pooling weight on the exact nested posterior: 0 below ESS 20, 1 above ESS 200 (log-linear)
EIG_QUAD_DIMS = 36        # quadratic classifier terms use the leading principal components of the latent means
GOAL_BODIES_POOL, GOAL_BODIES_OUTER = 4, 32  # simulated bodies per draw for the success probability p(goal | draw)
GOAL_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _eig_spec_check(spec):
    """Same keys and checks as p_meets_spec; unknown keys raise (a misspelled key must not drop a requirement)."""
    v = globals().get("_validate_spec")
    if v is not None:
        return v(spec)
    allowed = ("Hci_kOe_min", "BHmax_MGOe_min", "Br_kGs_min", "density_rel_min", "T_operating_C")
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"spec must be a non-empty dict with keys from {allowed}")
    bad = sorted(k for k in spec if k not in allowed)
    if bad:
        raise ValueError(f"unknown spec keys {bad}; allowed: {list(allowed)}")
    for k, val in spec.items():
        if not _is_number(val):
            raise ValueError(f"spec[{k!r}] must be a number")
    return spec


def _lbfgs(fg, x0, iters=200, m=10, tol=1e-8):
    """Minimise a smooth convex f given fg(x) -> (f, grad): two-loop L-BFGS with a backtracking Armijo search."""
    x = np.array(x0, float)
    f, g = fg(x)
    S, Y = [], []
    for _ in range(iters):
        q = g.copy()
        hist = []
        for s, y in reversed(list(zip(S, Y))):
            r = 1.0 / (y @ s)
            a = r * (s @ q)
            q -= a * y
            hist.append((r, a, s, y))
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for r, a, s, y in reversed(hist):
            q += s * (a - r * (y @ q))
        d = -q
        gd = g @ d
        if not gd < 0:
            d, gd, S, Y = -g, -(g @ g), [], []
        t = 1.0
        while True:
            xn = x + t * d
            fn, gn = fg(xn)
            if fn <= f + 1e-4 * t * gd or t < 1e-12:
                break
            t *= 0.5
        s, y = xn - x, gn - g
        done = abs(f - fn) <= tol * max(1.0, abs(f))
        if y @ s > 1e-14:
            S.append(s)
            Y.append(y)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        x, f, g = xn, fn, gn
        if done:
            break
    return x


class _Features:
    """Classifier / regression inputs: standardised summaries (clipped at 6 sd) and quadratic terms of the leading
    EIG_QUAD_DIMS principal components of the first n_quad columns (the per-latent means). Fitted on training rows."""

    def __init__(self, X, n_quad):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.clip((X - self.mu) / self.sd, -6.0, 6.0)[:, :n_quad]
        k = min(EIG_QUAD_DIMS, Z.shape[1])
        if k:
            lam, V = np.linalg.eigh(np.cov(Z.T).reshape(Z.shape[1], Z.shape[1]))
            self.V = V[:, ::-1][:, :k]
        else:
            self.V = np.zeros((Z.shape[1], 0))
        self.n_quad = n_quad

    def __call__(self, X):
        Z = np.clip((X - self.mu) / self.sd, -6.0, 6.0)
        P = Z[:, :self.n_quad] @ self.V
        k = P.shape[1]
        iu = np.triu_indices(k)
        quad = (P[:, :, None] * P[:, None, :])[:, iu[0], iu[1]] / math.sqrt(2.0) if k else np.zeros((len(Z), 0))
        return np.concatenate([Z, np.clip(quad, -36.0, 36.0)], 1)


class _Softmax:
    """Weighted multinomial logistic regression on _Features, fitted by L-BFGS with a small L2 penalty. Targets may
    be soft (rows of class probabilities)."""

    def __init__(self, X, T, sw, n_quad, l2=1e-4, iters=200):
        self.feat = _Features(X, n_quad)
        Phi = self.feat(X)
        n, p = Phi.shape
        K = T.shape[1]
        swn = sw / sw.sum()

        def fg(x):
            W, b = x[:p * K].reshape(p, K), x[p * K:]
            A = Phi @ W + b
            A -= A.max(1, keepdims=True)
            E = np.exp(A)
            Ssum = E.sum(1, keepdims=True)
            G = (E / Ssum - T) * swn[:, None]
            f = -float(np.sum(swn[:, None] * T * (A - np.log(Ssum)))) + 0.5 * l2 * float(np.sum(W * W))
            return f, np.concatenate([(Phi.T @ G + l2 * W).ravel(), G.sum(0)])
        x = _lbfgs(fg, np.zeros(p * K + K), iters=iters)
        self.W, self.b = x[:p * K].reshape(p, K), x[p * K:]

    def log_prob(self, X):
        A = self.feat(X) @ self.W + self.b
        return A - _lse(A, 1)[:, None]


def _ridge(X, t, sw, Xnew, n_quad, l2=1e-3):
    """Weighted ridge regression of t on [1, _Features(X)]; returns predictions at Xnew."""
    feat = _Features(X, n_quad)
    Phi = np.concatenate([np.ones((len(X), 1)), feat(X)], 1)
    w = sw / sw.sum()
    beta = np.linalg.solve(Phi.T @ (Phi * w[:, None]) + l2 * np.eye(Phi.shape[1]), Phi.T @ (w * t))
    return np.concatenate([np.ones((len(Xnew), 1)), feat(Xnew)], 1) @ beta


def _systematic(p, n, rng):
    """n indices drawn by systematic resampling with probabilities p (every index appears floor/ceil(n p) times)."""
    if n <= 0:
        return np.zeros(0, int)
    c = np.cumsum(p / p.sum())
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(c, u), len(p) - 1)


def _alloc(w, n, balanced):
    """Label counts summing to n: equal shares (balanced) or proportional to w (largest remainder)."""
    k = len(w)
    if balanced:
        c = np.full(k, n // k)
        c[: n - c.sum()] += 1
        return c
    raw = n * w / w.sum()
    c = np.floor(raw).astype(int)
    c[np.argsort(-(raw - c))[: n - c.sum()]] += 1
    return c


def _eig_halves(contrib, fold, offset):
    """Lower bound from each half of the simulated experiments and whether they agree within 2 combined SE."""
    est, var = [], []
    for f in (0, 1):
        c = contrib[fold == f]
        est.append(offset + float(c.mean()))
        var.append(float(c.var(ddof=1)) / len(c) if len(c) > 1 else float("inf"))
    return est, bool(abs(est[0] - est[1]) <= 2.0 * math.sqrt(var[0] + var[1]))


def _std_normal_cdf(z):
    """Phi(z), accurate in both tails (numpy has no erfc)."""
    z = np.asarray(z, float)
    return 0.5 * np.frompyfunc(math.erfc, 1, 1)(-z / math.sqrt(2.0)).astype(float)


def _bern_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


# ----------------------------------------------------------------------------------------------------------------
class PhysicsModel:
    def __init__(self, ph, fw, posterior_path, hashes, verified=False, weights=None, ms=None, link=None, an=None,
                 anneal_posterior=None):
        self.ph, self.fw, self.hashes, self.verified = ph, fw, hashes, verified
        self.ms, self.link = ms, link  # C6: microstructure link functions; coercivity link (work package 1) or None
        # C7 anneal layer: mechanism weights and stored draws (posterior and prior) per mechanism, kept apart from
        # every C5/C6 array; drawn with their own generator so no C5/C6 draw of a call changes
        self.an, self.an_post, self.an_prior, self.an_meta, self.an_dw = an, {}, {}, None, {}
        if an is not None:
            za = np.load(anneal_posterior, allow_pickle=False)
            self.an_meta = json.loads(str(za["__meta__"]))
            for key in za.files:
                if key == "__meta__":
                    continue
                grp, mech, par = key.split("__", 2)
                (self.an_post if grp == "post" else self.an_prior).setdefault(mech, {})[par] = np.asarray(za[key], float)
        z = np.load(posterior_path, allow_pickle=False)
        self.meta = json.loads(str(z["__meta__"]))
        self.post, self.prior = {}, {}
        # C6: the c6_* per-draw parameters are kept apart from the C5 parameters, so every C5 computation (draw
        # assignment, copies of resampled draws, conditioning) sees exactly the C5 arrays
        self.c6post, self.c6prior = {}, {}
        for key in z.files:
            if key == "__meta__":
                continue
            grp, mech, par = key.split("__", 2)
            if par.startswith("c6_"):
                tgt = self.c6post if grp == "post" else self.c6prior
            else:
                tgt = self.post if grp == "post" else self.prior
            tgt.setdefault(mech, {})[par] = np.asarray(z[key], dtype=float)
        # every indicator's weights (density, magnetic and, from v4, load_timing)
        self.w = {k: dict(v) for k, v in self.meta["weights"].items()}
        # per-draw load-timing probabilities (v4): stored as post__<H>__ltp_<label>, kept apart from the parameters
        self.ltp = {}
        for m, v in self.post.items():
            if all(f"ltp_{k}" in v for k in LT_LEVELS):
                P = np.stack([v.pop(f"ltp_{k}") for k in LT_LEVELS], 1)
                self.ltp[m] = P / P.sum(1, keepdims=True)
        self.dw = weights or {m: np.ones(len(next(iter(v.values())))) for m, v in self.post.items()}
        self.preps = self.meta["density_preps"]
        self.conditioning = None

    def _stamp(self, res):
        res.update(tool_version=TOOL_VERSION, evidence_class=EVIDENCE_CLASS, artifact_sha256=self.hashes,
                   artifact_verified=self.verified)
        if self.conditioning is not None:
            res["conditioned_on"] = self.conditioning
        return res

    # ---------------------------------------------------------------- routes and hulls
    def _route(self, route):
        """Resolved route. C7: every key must be a route key, an accepted alias (ROUTE_ALIASES, applied and recorded) or
        an informational key (echoed, never read); anything else raises with the closest route keys, so a misspelled
        key can no longer leave a calculation at its default. alignment_cos and alignment_cos_theta are one physical
        <cos theta> (0.5-0.95) that feeds both magnetic paths; giving both with different values raises. The 900 C
        hold (anneal_hold_h) is resolved for annealed routes. r["_resolution"] records what was applied."""
        given = dict(route or {})
        res = {"aliases_applied": {}, "informational_keys": [], "alignment": None, "anneal_hold": None}
        for alias, key in ROUTE_ALIASES.items():
            if alias in given:
                v = given.pop(alias)
                if key in given and given[key] != v:
                    raise ValueError(f"{alias} is an alias of {key}; they were given different values "
                                     f"({v!r} and {given[key]!r})")
                given[key] = v
                res["aliases_applied"][alias] = key
        known = set(ROUTE_DEFAULTS) | {ALIGN_KEY, ANNEAL_HOLD_KEY}
        unknown = sorted(str(k) for k in given
                         if k not in known and k not in INFORMATIONAL_KEYS and not str(k).startswith("_"))
        if unknown:
            import difflib
            hints = {k: difflib.get_close_matches(k, sorted(known | set(ROUTE_ALIASES)), n=2, cutoff=0.5)
                     for k in unknown}
            raise ValueError(f"unknown route keys {unknown} (closest route keys: {hints}); route keys: "
                             f"{sorted(known)}; aliases: {ROUTE_ALIASES}; informational (echoed, not read): "
                             f"{list(INFORMATIONAL_KEYS)}")
        res["informational_keys"] = sorted(k for k in given if k in INFORMATIONAL_KEYS)
        a_c5, a_link = given.get("alignment_cos"), given.get(ALIGN_KEY)
        if a_c5 is not None and a_link is not None and not (
                _is_number(a_c5) and _is_number(a_link) and float(a_c5) == float(a_link)):
            raise ValueError(f"alignment_cos ({a_c5!r}) and {ALIGN_KEY} ({a_link!r}) are the same physical <cos theta> "
                             "in C7 and feed both magnetic paths: give one, or equal values")
        a = a_link if a_link is not None else a_c5
        if a is not None:
            lo, hi = ALIGN_RANGE
            if not (_is_number(a) and lo <= float(a) <= hi):
                raise ValueError(f"alignment (<cos theta>) must be a number in [{lo:g}, {hi:g}] (0.5 = isotropic, every "
                                 "record to date; above 0.5 is a declared extrapolation: there is no aligned record)")
            given["alignment_cos"] = given[ALIGN_KEY] = float(a)
            res["alignment"] = {"value": float(a), "given_as": [k for k in ("alignment_cos", ALIGN_KEY)
                                                                if k in (route or {})],
                                "feeds": ["C5 remanence and the ceilings", "coercivity link (Br, Hci along-axis "
                                          "penalty, squareness)"]}
        r = dict(ROUTE_DEFAULTS)
        r.update(self.meta.get("route_defaults", {}))
        r.update(given)
        r[ALIGN_KEY] = float(r["alignment_cos"])
        if r["T_peak_C"] is None:
            raise ValueError("route needs T_peak_C")
        if r["pressure_mode"] not in PRESSURE_MODES:
            raise ValueError(f"pressure_mode must be one of {PRESSURE_MODES}")
        if r["anneal"] not in ANNEALS:
            raise ValueError(f"anneal must be one of {ANNEALS} (a 520 C step without the 900 C anneal is not modelled)")
        if r["current_path"] not in CURRENT_PATHS:
            raise ValueError(f"current_path must be one of {CURRENT_PATHS}")
        dc = r["degas_completeness"]
        if dc == "complete":
            pass
        elif _is_number(dc):
            if not 0.0 <= float(dc) <= 1.5:
                raise ValueError("degas_completeness must lie in [0, 1.5] (values above 1 are a declared extrapolation)")
            r["degas_completeness"] = float(dc)
        elif isinstance(dc, dict):
            kind = dc.get("distribution")
            if kind == "uniform":
                lo, hi = dc.get("low"), dc.get("high")
                if not (_is_number(lo) and _is_number(hi) and 0.0 <= float(lo) <= float(hi) <= 1.5):
                    raise ValueError("uniform degas_completeness needs numeric 0 <= low <= high <= 1.5")
            elif kind == "beta":
                a, b = dc.get("a"), dc.get("b")
                if not (_is_number(a) and _is_number(b) and float(a) > 0 and float(b) > 0):
                    raise ValueError("beta degas_completeness needs numeric a > 0 and b > 0")
            else:
                raise ValueError("degas_completeness dict needs distribution 'uniform' (low, high) or 'beta' (a, b)")
        else:
            raise ValueError("degas_completeness must be a number in [0, 1.5], a distribution dict "
                             "({'distribution': 'uniform', 'low', 'high'} or {'distribution': 'beta', 'a', 'b'}) "
                             "or 'complete'")
        fd = r[FEEDSTOCK_KEY]
        lo, hi = FEEDSTOCK_RANGE_UM
        if not (_is_number(fd) and lo <= float(fd) <= hi):
            raise ValueError(f"{FEEDSTOCK_KEY} must be a number in [{lo:g}, {hi:g}] (um; {FEEDSTOCK_REF_UM:g} = the "
                             "fitted feedstock; other values are a declared, uncalibrated extrapolation)")
        r[FEEDSTOCK_KEY] = float(fd)
        if r[FEEDSTOCK_KEY] != FEEDSTOCK_REF_UM and r["prep"] in self.preps:
            raise ValueError(f"fitted preparation {r['prep']!r} is the {FEEDSTOCK_REF_UM:g} um feedstock: a body made "
                             f"from other feedstock needs prep 'new' or another new label")
        hold = r.get(ANNEAL_HOLD_KEY)
        if r["anneal"] == "none":
            if hold is not None:
                raise ValueError(f"{ANNEAL_HOLD_KEY} applies to an annealed route (anneal '900' or '900+520')")
            r.pop(ANNEAL_HOLD_KEY, None)
        else:
            lo, hi = ANNEAL_HOLD_RANGE
            if hold is None:
                hold = ANNEAL_HOLD_DEFAULT
                res["anneal_hold"] = (f"default {ANNEAL_HOLD_DEFAULT:g} h at 900 C: the planned program (September "
                                      "timeline), not confirmed for #8")
            elif hold == ANNEAL_HOLD_AS8:
                res["anneal_hold"] = ("as_#8: the same unconfirmed program as #8 (densification dose identified by "
                                      "#8 per anneal mechanism; grain growth over the declared #8 hold scenarios)")
            elif _is_number(hold) and lo <= float(hold) <= hi:
                hold = float(hold)
                res["anneal_hold"] = f"{hold:g} h at 900 C (stated)"
            else:
                raise ValueError(f"{ANNEAL_HOLD_KEY} must be a number of hours in [{lo:g}, {hi:g}] or "
                                 f"{ANNEAL_HOLD_AS8!r}")
            r[ANNEAL_HOLD_KEY] = hold
        r["_resolution"] = res
        return r

    @staticmethod
    def _alignment(r):
        a = float(r.get(ALIGN_KEY, ALIGN_DEFAULT))
        return {ALIGN_KEY: a, "declared_extrapolation": bool(a > ALIGN_DEFAULT),
                "note": ("C7: alignment_cos and alignment_cos_theta are one <cos theta>. In the coercivity link Br scales "
                         "with it, Hci takes the declared along-axis penalty and Hk/Hci rises toward an aligned "
                         "magnet's; in C5 Br scales with it but Hci and squareness keep their isotropic fits, so C5's "
                         "(BH)max above 0.5 is not an aligned-magnet estimate. "
                         + ("Above 0.5 is a declared extrapolation: there is no aligned record."
                            if a > ALIGN_DEFAULT else "0.5 = isotropic, every record to date."))}

    @staticmethod
    def _feedstock_declared(r):
        return float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM)) != FEEDSTOCK_REF_UM

    def _feedstock_info(self, r):
        dec = self._feedstock_declared(r)
        return {FEEDSTOCK_KEY: float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM)), "declared": dec,
                "status": STATUS_DECLARED if dec else STATUS_FITTED,
                "note": ("declared feedstock particle size: grain size, an empirical grain-size factor on every "
                         "mechanism's coercivity, an oxidation penalty, a residual-hydrogen term and the density layer "
                         "follow uncalibrated priors (README, 'Feedstock particle size'); no record constrains them")
                if dec else
                        "the fitted feedstock (about 4 um); the feedstock layer is inactive"}

    @staticmethod
    def _completeness_is_default(r):
        dc = r["degas_completeness"]
        return _is_number(dc) and float(dc) == 1.0

    @staticmethod
    def _sample_completeness(dc, size, rng):
        """Draws of the degassing completeness d (fraction of the fitted D/E state; > 1 = declared extrapolation).
        "complete" returns 1 (the generic degassing terms are held at the D/E state; hydrogen is zeroed separately)."""
        if dc == "complete":
            return np.ones(size)
        if _is_number(dc):
            return np.full(size, float(dc))
        if dc["distribution"] == "uniform":
            return rng.uniform(float(dc["low"]), float(dc["high"]), size)
        return rng.beta(float(dc["a"]), float(dc["b"]), size)

    def _expected_state(self, r):
        """Expected degassing state E[occurrence x min(d, 1)], used only for the hull distance."""
        dc = r["degas_completeness"]
        if dc == "complete":
            m = 1.0
        elif _is_number(dc):
            m = min(float(dc), 1.0)
        else:
            m = float(np.mean(np.minimum(self._sample_completeness(dc, 4000, np.random.default_rng(0)), 1.0)))
        return float(r["degas"]) * m

    def _degas_state(self, r, occ_uniform, rng_extra):
        """Per-draw degassing state (occurrence x completeness) and the extrapolation multiplier eps (only used where
        the state exceeds 1). occ_uniform: the uniforms v3 already drew for occurrence, so default inputs reproduce v3
        exactly; rng_extra is consulted only for non-default completeness."""
        occ = (occ_uniform < float(r["degas"])).astype(float)
        if self._completeness_is_default(r) or r["degas_completeness"] == "complete":
            return occ, None
        d = self._sample_completeness(r["degas_completeness"], occ_uniform.shape, rng_extra)
        eps = rng_extra.uniform(0.0, 1.0, occ_uniform.shape)
        return occ * d, eps

    def _arrays(self, r):
        R = self.ph.Route(T_peak_C=float(r["T_peak_C"]), heating_rate_C_per_min=float(r["heating_rate_C_per_min"]),
                          dwell_min=float(r["dwell_min"]), P_MPa=float(r["P_MPa"]), pressure_mode=r["pressure_mode"],
                          cooling_rate_C_per_min=float(r["cooling_rate_C_per_min"]),
                          load_ramp_min=float(r["load_ramp_min"]), degas=1.0, anneal=r["anneal"],
                          alignment_cos=float(r["alignment_cos"]), T_measure_C=float(r["T_measure_C"]),
                          current_path=r["current_path"])
        S = self.ph.stack([self.ph.schedule(R)])
        return dict(S=S, T_peak=np.array([R.T_peak_C]), rate=np.array([R.heating_rate_C_per_min]),
                    dwell=np.array([R.dwell_min]), P=np.array([R.P_MPa]),
                    cooling=np.array([1.0 if R.pressure_mode == "cooling" else 0.0]), degas=np.array([1.0]),
                    mode_dwell=np.array([1.0 if R.pressure_mode == "dwell" else 0.0]),
                    path=np.array([0.0 if R.current_path == "insulated" else 1.0]), hfac=np.array([1.0]),
                    a900=np.array([1.0 if R.anneal in ("900", "900+520") else 0.0]),
                    a520=np.array([1.0 if R.anneal == "900+520" else 0.0]), align=np.array([R.alignment_cos]),
                    Tmeas=np.array([R.T_measure_C]))

    def hull(self, r, rho=None):
        """Nearest-neighbour distance to the fitted routes in standardised process space; out_of_hull if the distance
        exceeds 1 or a feature leaves the fitted range by more than a quarter of its scale. The magnetic distance is the
        one H0's discrepancy uses (forward.mag_hull_distance)."""
        out = {}
        for module in ("density", "magnetic"):
            H = self.meta["hull"][module]
            feats = H["features"]
            x = self._features(r, rho, feats)
            pts = np.asarray(H["points"], float)
            sc = np.asarray(H["scales"], float)
            d = float(np.sqrt((((pts - x) / sc) ** 2).sum(1)).min())
            lo, hi = pts.min(0), pts.max(0)
            outside = [f for f, xi, l, h, s in zip(feats, x, lo, hi, sc) if xi < l - 0.25 * s or xi > h + 0.25 * s]
            if r["pressure_mode"] == "dwell":  # declared scenario: no fitted record has load from the dwell start
                outside.append("pressure_mode=dwell")
            if module == "magnetic" and r.get("current_path") == "insulated":
                outside.append("current_path=insulated")
            if module == "magnetic" and r.get("degas_completeness") == "complete":
                outside.append("degas_completeness=complete")
            if self._feedstock_declared(r):  # C4: no record was made from other feedstock
                outside.append(f"{FEEDSTOCK_KEY}={float(r[FEEDSTOCK_KEY]):g}")
            out[module] = dict(distance=d, out_of_hull=bool(d > 1.0 or outside), outside_features=outside)
        return out

    @staticmethod
    def _features(r, rho, feats):
        val = dict(T_peak_C=float(r["T_peak_C"]), log10_rate=math.log10(float(r["heating_rate_C_per_min"])),
                   dwell_min=float(r["dwell_min"]), P_MPa=float(r["P_MPa"]),
                   load_on_cooling=_load_on_cooling(r),
                   degas=float(r.get("_degas_expected", r["degas"])), rel_density=float(rho if rho is not None else 0.98))
        return np.array([val[f] for f in feats], float)

    def _hull_distance(self, r, rho):
        d = self.fw.mag_hull_distance(float(r["T_peak_C"]), float(rho), _load_on_cooling(r),
                                      float(r.get("_degas_expected", r["degas"])),
                                      float(r["heating_rate_C_per_min"]))
        if self._feedstock_declared(r):
            # C4 (declared): every record used the 4 um feedstock, so a halving or doubling of the feedstock size adds
            # one standardised unit (as 100 C does) to H0's nearest-neighbour distance
            d = math.sqrt(d * d + (math.log(float(r[FEEDSTOCK_KEY]) / FEEDSTOCK_REF_UM) / math.log(2.0)) ** 2)
        return d

    # ---------------------------------------------------------------- draw assignment
    def _mechs(self, kind, group="post"):
        src = self.post if group == "post" else self.prior
        return [m for m in self.w[kind] if m in src]

    def _assign_bma(self, kind, n, rng, group="post"):
        """n draws from the model average: mechanism labels by weight, parameter indices by the per-draw weights."""
        src = self.post if group == "post" else self.prior
        mechs = self._mechs(kind, group)
        wm = np.array([self.w[kind][m] for m in mechs], float) if group == "post" else np.ones(len(mechs))
        counts = rng.multinomial(n, wm / wm.sum())
        labels, idx = [], []
        for m, c in zip(mechs, counts):
            if c == 0:
                continue
            K = len(next(iter(src[m].values())))
            p = self.dw[m] / self.dw[m].sum() if group == "post" else np.full(K, 1.0 / K)
            labels += [m] * c
            idx.append(rng.choice(K, size=c, p=p))
        labels, idx = np.array(labels), np.concatenate(idx)
        perm = rng.permutation(n)
        return labels[perm], idx[perm]

    def _assign_strata(self, kind, n_per, rng):
        """n_per draws from every mechanism with non-zero weight (stratified inner sets for eig)."""
        labels, idx = [], []
        for m in self._mechs(kind):
            if self.w[kind][m] <= 0:
                continue
            K = len(next(iter(self.post[m].values())))
            p = self.dw[m] / self.dw[m].sum()
            labels += [m] * n_per
            idx.append(rng.choice(K, size=n_per, p=p))
        return np.array(labels), np.concatenate(idx)

    def _params(self, src, m, idx):
        return {k: v[idx][:, None] for k, v in src[m].items()}

    # ---------------------------------------------------------------- conditional indicators (v4: load timing)
    def _indicators(self):
        """Indicators whose labels are drawn conditionally on the magnetic draw (no parameter arrays of their own)."""
        return ["load_timing"] if self.ltp else []

    def _assign_indicator(self, name, mlab, midx, rng, group="post"):
        """Per-draw labels of a conditional indicator, aligned with (mlab, midx): load_timing labels are drawn from
        the draw's own posterior probabilities p(LT | theta, data) (posterior group) or uniformly (prior group)."""
        if name != "load_timing":
            raise ValueError(f"unknown conditional indicator {name!r}")
        n = len(mlab)
        u = rng.random(n)
        out = np.empty(n, dtype=object)
        for m in np.unique(mlab):
            pos = np.where(mlab == m)[0]
            if group == "post" and m in self.ltp:
                P = self.ltp[m][midx[pos]]
            else:
                P = np.full((len(pos), len(LT_LEVELS)), 1.0 / len(LT_LEVELS))
            c = np.cumsum(P, 1)
            k = np.minimum((u[pos][:, None] > c).sum(1), len(LT_LEVELS) - 1)
            out[pos] = np.array(LT_LEVELS, dtype=object)[k]
        return out.astype(str)

    def _assign_strata_indicator(self, name, n_per, rng):
        """n_per joint draws (mlab, midx, labels) per level of a conditional indicator: (m, i) sampled with
        probability proportional to w_m * dw_m[i] / sum(dw_m) * p_m[i, level] (exactly p(m, i | level))."""
        if name != "load_timing":
            raise ValueError(f"unknown conditional indicator {name!r}")
        mechs = [m for m in self._mechs("magnetic") if self.w["magnetic"][m] > 0]
        blocks = []
        for m in mechs:
            base = self.w["magnetic"][m] * self.dw[m] / self.dw[m].sum()
            blocks.append(base[:, None] * self.ltp[m])
        Wt = np.concatenate(blocks, 0)
        offs = np.cumsum([0] + [len(b) for b in blocks])
        labs, idxs, lts = [], [], []
        for k, lvl in enumerate(LT_LEVELS):
            pk = Wt[:, k] / Wt[:, k].sum()
            j = rng.choice(len(pk), size=n_per, p=pk)
            mi = np.searchsorted(offs, j, side="right") - 1
            labs.append(np.array(mechs, dtype=object)[mi])
            idxs.append(j - offs[mi])
            lts.append(np.full(n_per, lvl, dtype=object))
        return (np.concatenate(labs).astype(str), np.concatenate(idxs).astype(int),
                np.concatenate(lts).astype(str))

    # ---------------------------------------------------------------- per-draw latent quantities at a route
    def _dens_mu(self, p, m, A, off, r):
        """Route-level relative density of density mechanism m for draws p (values (K, 1)) with preparation offset off.
        At the fitted 4 um feedstock this is forward.density exactly; for a declared feedstock the C5 density layer
        (declared, uncalibrated) adds its ln-rate offset and green-density change."""
        fd = float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM))
        if fd == FEEDSTOCK_REF_UM:
            return self.fw.density(p, m, A, lnk_offset=off, xp=np)
        miss = [k for k in C5_DENSITY_PARAMS if k not in p]
        if miss:
            raise ValueError(f"this posterior carries no density-layer draws {miss}: {FEEDSTOCK_KEY} other than "
                             f"{FEEDSTOCK_REF_UM:g} needs a package that declares the C5 density layer")
        q = dict(p, phi0=self.fw.feedstock_green_phi0(p, fd, np))
        mu = self.fw.density(q, m, A, lnk_offset=off + self.fw.feedstock_density_offset(p, fd, np), xp=np)
        return self.fw.feedstock_density_discrepancy(p, mu, fd, np)

    def _density_mean(self, A, r, dlab, didx, prep_z, group="post"):
        """Per-draw route-level relative density (before specimen scatter) and its scatter scale."""
        src = self.post if group == "post" else self.prior
        n = len(dlab)
        mu, sig = np.empty(n), np.empty(n)
        for m in np.unique(dlab):
            pos = np.where(dlab == m)[0]
            p = self._params(src, m, didx[pos])
            if r["prep"] in self.preps and group == "post":
                off = p["tau_prep"] * p[f"z_prep_{self.preps.index(r['prep'])}"]
            else:
                off = p["tau_prep"] * prep_z[pos]
            mu[pos] = self._dens_mu(p, m, A, off, r)[:, 0]
            sig[pos] = p["sig_rho"][:, 0]
        return mu, sig

    def _magnetic_latent(self, A, r, mlab, midx, rho, degas_ind, z_hull, u_spec=None, group="post", eps=None,
                         indicators=None, sem=False):
        """Per-draw latent magnetic quantities. rho, degas_ind, z_hull, u_spec: (n,) arrays aligned with the draws.
        ln Hci includes u_spec (the body's specimen effect) when given; otherwise it is the route-level value.
        degas_ind is the continuous degassing state (occurrence x completeness). Where it exceeds 1 (a declared
        extrapolation beyond the fitted D/E state) the forward model is evaluated at 1 and ln Hci gains
        eps * (state - 1) * [ln Hci(state 1) - ln Hci(state 0)] for the same draw, eps ~ U(0, 1); the residual-hydrogen
        index is reduced by the factor 1 - eps * (state - 1). With degas_completeness "complete" a degassed draw has
        state 1 for the generic degassing terms and no hydrogen (H1 and the trap alternative see h0 = 0).
        indicators: {"load_timing": labels (n,)} aligned with the draws (required when the posterior carries LT; the
        same labels must be used for every route of a joint draw). lh_res is the physical residual-hydrogen index of
        the sintered body (pore closure by the load mode), floored at the declared detection limit H_DETECTION_INDEX.
        sem=True (C5) also returns the model grain of the body at each stage of SEM_STAGES (g_as_sintered, g_900+520),
        the BSE continuity index W_bse at this route (the draw's own load-timing penalty and degassing state) and the
        draw's declared C5 measurement parameters; every other key is computed exactly as without it."""
        src = self.post if group == "post" else self.prior
        n = len(mlab)
        if self.ltp:
            if not indicators or "load_timing" not in indicators:
                raise ValueError("this posterior carries the load-timing indicator: pass "
                                 "indicators={'load_timing': labels} from _assign_indicator")
            ltlab = np.asarray(indicators["load_timing"]).astype(str)
            if ltlab.shape != (n,):
                raise ValueError("load_timing labels must be aligned with the draws")
        else:
            ltlab = np.full(n, "load")
        complete = r.get("degas_completeness") == "complete"
        fd = float(r.get(FEEDSTOCK_KEY, FEEDSTOCK_REF_UM))
        milled = fd != FEEDSTOCK_REF_UM  # C4 feedstock layer; inactive (code path unchanged) at the fitted feedstock
        keys = ("lh", "lbr", "sq", "lgrain", "lh_res", "sig_spec", "sig_piece", "sig_Br", "sig_sq", "Js", "HA", "nB",
                "dB", "HR", "kR", "grain")
        if sem:
            keys = keys + tuple("g_" + s for s in SEM_STAGES) + ("W_bse",) + C5_SEM_PARAMS + ("ln_chem_offset",)
        out = {k: np.empty(n) for k in keys}
        for m in np.unique(mlab):
            pos = np.where(mlab == m)[0]
            p = self._params(src, m, midx[pos])
            rr = rho[pos][:, None]
            state = degas_ind[pos][:, None]
            over = np.maximum(state - 1.0, 0.0)
            Ad = dict(A)
            Ad["degas"] = np.minimum(state, 1.0)
            Ad["hfac"] = np.where(state >= 1.0, 0.0, 1.0) if complete else np.ones_like(state)
            if milled:
                miss = [k for k in FEEDSTOCK_PARAMS if k not in p]
                if miss:
                    raise ValueError(f"this posterior carries no feedstock-layer draws {miss}: {FEEDSTOCK_KEY} other "
                                     f"than {FEEDSTOCK_REF_UM:g} needs a package that declares the layer")
                # residual hydrogen of the milled powder, as the current-feedstock degassing state with the same index
                # (no hydrogen, hence no change, where degassing is complete: hfac = 0)
                Ad["degas"] = np.where(Ad["hfac"] > 0, self.fw.feedstock_degas_equivalent(p, Ad["degas"], fd, np),
                                       Ad["degas"])
                # sintered grain from the declared feedstock vs the same route and draw from the 4 um feedstock; one
                # empirical grain-size factor for every mechanism. Ad carries no grain hook, so H4's Kronmueller term
                # stays at g_ref and the feedstock change is counted once.
                g_fd = self.fw.grain(p, dict(A, fd_d0_factor=self.fw.feedstock_d0_factor(p, fd, np)), xp=np)
                g_ref = self.fw.grain(p, A, xp=np)
                fshift = (np.log(self.fw.feedstock_grain_factor(p, g_fd, g_ref, np))
                          - self.fw.feedstock_oxidation_penalty(p, fd, np))
            hd = np.array([self._hull_distance(r, x) for x in rho[pos]])[:, None] if m == "H0" else 0.0
            u = u_spec[pos][:, None] if u_spec is not None else 0.0
            zh = z_hull[pos][:, None]
            lt = ltlab[pos]
            if sem:  # C6 (coercivity link v2): C5 chemistry/porosity offset vs the D/E reference state; no randomness
                e_c = eps[pos][:, None] if np.any(over > 0) else 0.0
                out["ln_chem_offset"][pos] = self._c6_chem_offset(p, m, A, Ad, rr, hd, zh, lt, fd, milled, over, e_c)
            if np.any(over > 0):
                e = eps[pos][:, None]
                raw = self.fw._log_hci_raw(p, m, Ad, rr, u, hd, zh, np, lt)
                one = dict(Ad, degas=np.ones_like(state), hfac=np.ones_like(state))
                zero = dict(Ad, degas=np.zeros_like(state), hfac=np.ones_like(state))
                slope = (self.fw._log_hci_raw(p, m, one, rr, 0.0, hd, zh, np, lt)
                         - self.fw._log_hci_raw(p, m, zero, rr, 0.0, hd, zh, np, lt))
                tot = raw + e * over * slope
                lh = self.fw.cap_log_hci(tot + fshift if milled else tot, np)
            else:
                raw = self.fw._log_hci_raw(p, m, Ad, rr, u, hd, zh, np, lt)
                lh = self.fw.cap_log_hci(raw + fshift if milled else raw, np)
            out["lh"][pos] = lh[:, 0]
            out["lbr"][pos] = np.log(self.fw.remanence(p, A, rr, np.exp(lh), xp=np))[:, 0]
            out["sq"][pos] = (self.fw.squareness_logit(p, A, xp=np) * np.ones((len(pos), 1)))[:, 0]
            g = (g_fd if milled else self.fw.grain(p, A, xp=np))[:, 0]
            out["grain"][pos], out["lgrain"][pos] = g, np.log(g)
            hres = self.ph.desorption(A["S"], A["cooling"], Ad["degas"], p, mode_dwell=A.get("mode_dwell")) * Ad["hfac"]
            if np.any(over > 0):
                hres = hres * np.maximum(1.0 - eps[pos][:, None] * over, 1e-6)
            out["lh_res"][pos] = np.log(np.maximum(hres, H_DETECTION_INDEX))[:, 0]
            for k in ("sig_spec", "sig_piece", "sig_Br", "sig_sq", "Js", "HR", "kR"):
                out[k][pos] = p[k][:, 0]
            for k, default in (("HA", 75.0), ("nB", 0.17), ("dB", 0.004)):
                out[k][pos] = p[k][:, 0] if k in p else default
            if sem:  # C5 (declared): both stages of the same piece and the BSE continuity index; no randomness
                miss = [k for k in C5_SEM_PARAMS if k not in p] + ([] if "Tw" in p or "c5_Tw" in p else ["c5_Tw"])
                if miss:
                    raise ValueError(f"this posterior carries no C5 SEM/BSE draws {miss}: the SEM/BSE outputs need a "
                                     "package that declares them")
                d0f = self.fw.feedstock_d0_factor(p, fd, np) if milled else None
                for s in SEM_STAGES:
                    out["g_" + s][pos] = self.fw.grain_stage(p, A, s, d0f, np)[:, 0]
                pen = self.fw.lt_penalties(p, Ad, np)
                lab = np.asarray(lt).astype(str)
                sel = sum(np.where(lab == k, 1.0, 0.0)[:, None] * pen[k] for k in LT_LEVELS)
                W = self.fw.bse_continuity(p, A, sel, fd if milled else None, np)
                out["W_bse"][pos] = np.broadcast_to(W, (len(pos), 1))[:, 0]
                for k in C5_SEM_PARAMS:
                    out[k][pos] = p[k][:, 0]
        return out

    def _c6_chem_offset(self, p, m, A, Ad, rr, hd, zh, lt, fd, milled, over, e):
        """C6 (CHEMISTRY_COUPLING.md, micromagnetics session, Oct 18): ln Hci of the draw's actual chemistry minus the
        D/E reference chemistry (completeness 1, 4 um hydrogen, rho >= E's), same draw, mechanism, route and load-timing
        label, so temperature, dwell, anneal, load timing and H0 cancel. Includes the fine-feedstock oxidation penalty,
        the milled-powder hydrogen, degassing completeness and over-completion, and open porosity (H5). Excludes the
        feedstock grain factor (the link has its own grain physics). Exactly 0 at historical D's and E's routes."""
        fw = self.fw
        act = fw._log_hci_raw(p, m, Ad, rr, 0.0, hd, zh, np, lt)
        if np.any(over > 0):
            one = dict(Ad, degas=np.ones_like(Ad["degas"]), hfac=np.ones_like(Ad["degas"]))
            zero = dict(Ad, degas=np.zeros_like(Ad["degas"]), hfac=np.ones_like(Ad["degas"]))
            act = act + e * over * (fw._log_hci_raw(p, m, one, rr, 0.0, hd, zh, np, lt)
                                    - fw._log_hci_raw(p, m, zero, rr, 0.0, hd, zh, np, lt))
        if milled:
            act = act - fw.feedstock_oxidation_penalty(p, fd, np)
        ref_A = dict(A, degas=np.ones_like(Ad["degas"]), hfac=np.ones_like(Ad["degas"]))
        ref = fw._log_hci_raw(p, m, ref_A, np.maximum(rr, fw.E_RHO), 0.0, hd, zh, np, lt)
        return np.broadcast_to(act - ref, (len(rr), 1))[:, 0]

    # ---------------------------------------------------------------- C7 anneal layer
    def _has_c7(self):
        return self.an is not None and bool(self.an_post)

    def _c7_draws(self, n, rng, group="post"):
        """n anneal-layer draws: mechanism labels by the layer's weights (uniform for the prior group) and a stored draw
        of that mechanism (the stored posterior is resampled, so draws are uniform within a mechanism)."""
        src = self.an_post if group == "post" else self.an_prior
        mechs = [m for m in self.an.ANNEAL_MECHS if m in src]
        w = (np.array([self.an_meta["mechanism_weights"][m] for m in mechs], float) if group == "post"
             else np.ones(len(mechs)))
        lab = rng.choice(len(mechs), size=n, p=w / w.sum())
        p = {k: np.empty(n) for k in src[mechs[0]]}
        for j, m in enumerate(mechs):
            pos = np.where(lab == j)[0]
            if len(pos):
                K = len(next(iter(src[m].values())))
                dw = self.an_dw.get(m) if group == "post" else None   # C7.1: weights after condition_anneal()
                idx = (rng.integers(0, K, len(pos)) if dw is None else rng.choice(K, size=len(pos), p=dw / dw.sum()))
                for k in p:
                    p[k][pos] = src[m][k][idx]
        return np.array(mechs)[lab], p

    def _c7_hold(self, r, p):
        """Per-draw 900 C hold (h): the route's (default 2 h, the planned program) or, for 'as_#8', each draw's declared
        #8 hold scenario (the densification dose is #8's either way). An as-sintered route's paired anneal stages use
        the default."""
        hold = r.get(ANNEAL_HOLD_KEY, ANNEAL_HOLD_DEFAULT) if r["anneal"] != "none" else ANNEAL_HOLD_DEFAULT
        if hold == ANNEAL_HOLD_AS8:
            return np.asarray(p["c7_hold8_h"], float)
        return np.full(len(p["c7_hold8_h"]), float(hold))

    def _c7_grain_hold(self, lat, mlab, midx, group, hold, T_peak_C):
        """C5 model grain after 900 C for `hold` h (h per draw): the grain law's normal-growth term scales with the hold
        (D^3 linear in time; m = 3) and its abnormal-growth factor is the annealed one. Equals C5's
        g_after_900_520 at 2 h (the 520 C step adds no growth in C5)."""
        src = self.post if group == "post" else self.prior
        g_as, g_120 = np.asarray(lat["g_as_sintered"], float), np.asarray(lat["g_after_900_520"], float)
        out = np.empty(len(mlab))
        for m in np.unique(mlab):
            pos = np.where(mlab == m)[0]
            p = self._params(src, m, midx[pos])
            Aab, Tab = p["Aab"][:, 0], p["Tab"][:, 0]
            f_as = 1.0 + Aab * self.ph.sigmoid((float(T_peak_C) - Tab) / 15.0)
            f_an = 1.0 + Aab * self.ph.sigmoid((max(float(T_peak_C), 900.0) - Tab) / 15.0)
            n_as, n_120 = (g_as[pos] / f_as) ** 3, (g_120[pos] / f_an) ** 3
            out[pos] = f_an * np.maximum(n_as + (n_120 - n_as) * hold[pos] / 2.0, n_as) ** (1.0 / 3.0)
        return out

    def _c7_h4_hold_shift(self, r, mlab, midx, group, hold):
        """C7.1: change of C5's ln Hci for grain-size mechanism H4 draws when the 900 C hold is not the fitted 2 h: the
        Kronmueller ratio at the hold-scaled 4 um-feedstock grain over the ratio at the 2 h grain (H4's own grain basis;
        the feedstock factor enters every mechanism separately). Zero for other mechanisms, for as-sintered routes and at
        2 h. Non-grain anneal terms stay E's 2 h response (declared)."""
        out = np.zeros(len(mlab))
        if r["anneal"] == "none" or not np.any(mlab == "H4"):
            return out
        src = self.post if group == "post" else self.prior
        pos = np.where(mlab == "H4")[0]
        p = self._params(src, "H4", midx[pos])
        A = self._arrays(r)
        g_as = self.fw.grain_stage(p, A, "as_sintered", None, np)[:, 0]
        g_2h = self.fw.grain_stage(p, A, "after_900_520", None, np)[:, 0]
        T = float(r["T_peak_C"])
        Aab, Tab = p["Aab"][:, 0], p["Tab"][:, 0]
        f_as = 1.0 + Aab * self.ph.sigmoid((T - Tab) / 15.0)
        f_an = 1.0 + Aab * self.ph.sigmoid((max(T, 900.0) - Tab) / 15.0)
        n_as, n_2h = (g_as / f_as) ** 3, (g_2h / f_an) ** 3
        g_h = f_an * np.maximum(n_as + (n_2h - n_as) * np.asarray(hold, float)[pos] / 2.0, n_as) ** (1.0 / 3.0)
        DE = self.fw.grain(p, self.fw.E_ARR, xp=np)[:, 0]
        k = p["nB"][:, 0] * p["Js"][:, 0] / p["HKE"][:, 0]
        ratio = lambda D: 0.02 * np.logaddexp(0.0, (1.0 - k * np.log(D / DE)) / 0.02)
        out[pos] = np.log(np.maximum(ratio(g_h), 1e-6)) - np.log(np.maximum(ratio(g_2h), 1e-6))
        return out

    def _c7_apply_lh_shift(self, lh, shift):
        """Add a shift to C5's capped ln Hci under the same smooth ceiling (exact inverse of forward.cap_log_hci)."""
        if not np.any(shift):
            return lh
        L = math.log(self.fw.H_MAX)
        y = np.minimum(np.asarray(lh, float), L - 1e-9)
        raw = L - 0.25 * np.log(np.expm1(-4.0 * (y - L)))
        return np.where(shift != 0.0, self.fw.cap_log_hci(raw + shift, np), lh)

    def _c7_state(self, r, rho, lat, mlab, midx, group, rng, density_stage, draws=None):
        """Anneal-layer state of each draw's body: anneal mechanism and parameters, the 900 C hold, the body's density
        at every C6 stage, the route stage's density, historical E's gain under the same draw (900 C / 2 h), the factor
        on C5's Br at an annealed route, and (with the C5 SEM latents) the C5 model grain after the hold."""
        mech, p = draws if draws is not None else self._c7_draws(len(rho), rng, group)
        hold = self._c7_hold(r, p)
        an = self.an
        if density_stage == "route" and r["anneal"] != "none":   # density given at the route's stage: not inferred
            stage_rho = {s: np.asarray(rho, float) for s in C6_STAGES}
        else:   # C7.1: each stage from the as-sintered body through its whole executed sequence (CAPACITY's state)
            stage_rho = {"as_sintered": np.asarray(rho, float),
                         "after_900": an.anneal_density(p, mech, rho, [(900.0, hold)]),
                         "after_900_520": an.anneal_density(p, mech, rho, [(900.0, hold), SECOND_STEP])}
        route_stage = {"none": "as_sintered", "900": "after_900", "900+520": "after_900_520"}[r["anneal"]]
        e = np.full(len(rho), E_ROUTE_ANNEAL_RHO)
        gain_E = an.anneal_density(p, mech, e, [(900.0, np.full(len(rho), 2.0))]) / e
        br_factor = (stage_rho[route_stage] / np.asarray(rho, float)) / gain_E if route_stage != "as_sintered" else None
        out = {"mech": mech, "p": p, "hold_h": hold, "stage_rho": stage_rho, "route_stage": route_stage,
               "gain_E": gain_E, "br_factor": br_factor, "density_stage": density_stage,
               "h4_shift": self._c7_h4_hold_shift(r, mlab, midx, group, hold)}
        if lat is not None and "g_as_sintered" in lat:
            g_h = self._c7_grain_hold(lat, mlab, midx, group, hold, r["T_peak_C"])
            out["grain_stage"] = {"as_sintered": np.asarray(lat["g_as_sintered"], float), "after_900": g_h,
                                  "after_900_520": g_h}
        return out

    def _c7_block(self, r, c7, seed):
        """predict()'s C7 block: what the anneal layer did at this route, with the stage densities by mechanism and the
        readings a density instrument would return of them (archimedes, geometric)."""
        if c7 is None:
            return {"status": "anneal layer not loaded: annealed stages keep the as-sintered density (C6 behaviour)"}
        mech, p = c7["mech"], c7["p"]
        rng = np.random.default_rng([int(seed), 73])
        stages = {}
        for s, x in c7["stage_rho"].items():
            x = np.asarray(x, float)
            readings = {}
            for meth in ("archimedes", "geometric"):
                v, _ = self.an.read_density(p, x, meth, rng)
                readings[meth] = {k: float(q) * RHO_TD for k, q in zip(("q05", "median", "q95"),
                                                                        np.quantile(v, [0.05, 0.5, 0.95]))}
            stages[s] = {"density_g_cm3": {k: float(q) * RHO_TD for k, q in zip(
                ("q05", "q25", "median", "q75", "q95"), np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]))},
                         "by_anneal_mechanism_median_g_cm3": {m: float(np.median(x[mech == m])) * RHO_TD
                                                              for m in np.unique(mech)},
                         "reading_g_cm3": readings}
        hold = np.asarray(c7["hold_h"], float)
        return {"layer": self.an_meta.get("version"), "route_stage": c7["route_stage"],
                "density_stage_of_input": c7["density_stage"],
                "anneal_mechanism_weights": self.an_meta["mechanism_weights"],
                "anneal_hold_h": {"values": sorted({float(h) for h in np.unique(hold)}),
                                  "share": {f"{h:g}": float(np.mean(hold == h)) for h in np.unique(hold)}},
                "stages": stages,
                "E_reference_gain_median": float(np.median(c7["gain_E"])),
                "note": C7_STAGE_DENSITY_NOTE,
                "limits": ["the anneal layer is calibrated on one paired return (#8: 7.07 -> 7.58 g/cm3 after a 900 C "
                           "treatment whose hold, method and piece pairing are not stated); its mechanism weights are "
                           "prior-dominated between LPS, CLOSED and CAPACITY, which predict different responses for "
                           "bodies below about 90 %TD",
                           "magnetic anneal terms (C5's b_a1, b_a2, eta_a1, eta_a2, the link's re-wetting and 520 C "
                           "gain) are historical E's 2 h response at every hold",
                           "polished-section planarity, porosity and Nd-rich phase readings follow the as-sintered "
                           "density; the fracture-intercept feature factor and the link's Br follow the stage density",
                           "readings: archimedes counts c7_kopen of the open porosity as solid (protocol not stated by "
                           "RPI); geometric carries a relative bias c7_bgeo; neither includes piece-to-piece gradients"]}

    # ---------------------------------------------------------------- C7.1 anneal experiments: updating and selection
    ANNEAL_READINGS = ("archimedes", "archimedes_saturated", "geometric", "unspecified")

    def _an_pool(self):
        """Every stored anneal draw (posterior) as flat arrays: labels, parameters and weights (mechanism weight x
        per-draw weight)."""
        mechs = [m for m in self.an.ANNEAL_MECHS if m in self.an_post]
        lab, par, w = [], {}, []
        for m in mechs:
            K = len(next(iter(self.an_post[m].values())))
            dw = self.an_dw.get(m, np.ones(K))
            lab.append(np.full(K, m))
            w.append(self.an_meta["mechanism_weights"][m] * dw / dw.sum())
            for k, v in self.an_post[m].items():
                par.setdefault(k, []).append(v)
        return np.concatenate(lab), {k: np.concatenate(v) for k, v in par.items()}, np.concatenate(w)

    def _an_spec(self, spec, observed):
        """Validated anneal-experiment spec. start: {"g_cm3": x, "sd_rel": 0.03} (the body's true as-sintered density,
        e.g. from its released reading) or {"route": {...}} (a new body: the SPS density predictive); program: "as_#8"
        or a 900 C hold in hours; second_step: the 520 C / 2 h step (no densification); before / after: {"method":
        archimedes | geometric | unspecified} plus "g_cm3" for an observation (before may be omitted); same_piece:
        true, false or null (unknown, 50/50)."""
        if not isinstance(spec, dict):
            raise ValueError("an anneal experiment is a dict")
        allowed = {"name", "label", "start", "program", "second_step", "before", "after", "same_piece"}
        bad = sorted(set(spec) - allowed)
        if bad:
            raise ValueError(f"unknown anneal-experiment keys {bad}; allowed {sorted(allowed)}")
        st = spec.get("start")
        if not (isinstance(st, dict) and (("g_cm3" in st and _is_number(st["g_cm3"])) or isinstance(st.get("route"), dict))):
            raise ValueError("start must be {'g_cm3': number, 'sd_rel': optional} or {'route': {...}}")
        if isinstance(st.get("route"), dict) and self._route(st["route"])["anneal"] != "none":
            raise ValueError("start.route is the SPS route of an as-sintered body (anneal 'none')")
        prog = spec.get("program", ANNEAL_HOLD_AS8)
        lo, hi = ANNEAL_HOLD_RANGE
        if prog != ANNEAL_HOLD_AS8 and not (_is_number(prog) and lo <= float(prog) <= hi):
            raise ValueError(f"program must be {ANNEAL_HOLD_AS8!r} or a 900 C hold in hours in [{lo:g}, {hi:g}]")
        for key in ("before", "after"):
            rd = spec.get(key)
            if rd is None:
                if key == "after":
                    raise ValueError("an anneal experiment needs an after reading")
                continue
            if not isinstance(rd, dict) or rd.get("method", "unspecified") not in self.ANNEAL_READINGS:
                raise ValueError(f"{key} needs a method in {self.ANNEAL_READINGS}")
            if observed and not _is_number(rd.get("g_cm3")):
                raise ValueError(f"an observation's {key} reading needs g_cm3")
        if spec.get("same_piece") not in (True, False, None):
            raise ValueError("same_piece must be true, false or null (unknown)")
        return spec

    def _an_readings(self, lab, par, spec, rng, start_draws=None):
        """Noiseless reading means and sds (relative) of the before/after readings for draws (lab, par), with the
        nuisances sampled once per draw: true start, piece offsets (shared when the same piece), reading methods (50/50
        when unspecified) and #8's hold scenario for "as_#8". Also returns the annealed true density of the piece."""
        an, n = self.an, len(lab)
        st = spec["start"]
        if "g_cm3" in st:
            mu = float(st["g_cm3"]) / RHO_TD
            rho_b = np.clip(mu * (1.0 + float(st.get("sd_rel", 0.03)) * rng.standard_normal(n)), 0.3, 1.0)
        else:
            rho_b = rng.choice(start_draws, size=n)
        same = spec.get("same_piece")
        same = rng.random(n) < 0.5 if same is None else np.full(n, bool(same))
        eta_b = par["c7_sdpiece"] * rng.standard_normal(n)
        eta_a = np.where(same, eta_b, par["c7_sdpiece"] * rng.standard_normal(n))
        prog = spec.get("program", ANNEAL_HOLD_AS8)
        hold = par["c7_hold8_h"] if prog == ANNEAL_HOLD_AS8 else np.full(n, float(prog))
        steps = [(900.0, hold)] + ([SECOND_STEP] if spec.get("second_step") else [])
        a1 = an.anneal_density(par, lab, rho_b + eta_a, steps)
        out = {"rho_after_true": a1}
        for key, piece in (("before", rho_b + eta_b), ("after", a1)):
            rd = spec.get(key)
            if rd is None:
                continue
            m = rd.get("method", "unspecified")
            if m == "unspecified":   # a reading whose method is not stated: unsealed Archimedes or geometric, 50/50
                pick = rng.random(n) < 0.5
                mean = np.where(pick, an.reading_mean(par, piece, "archimedes"), an.reading_mean(par, piece, "geometric"))
                sd = np.where(pick, an.reading_sd(par, "archimedes"), an.reading_sd(par, "geometric"))
            else:
                mean, sd = an.reading_mean(par, piece, m), np.broadcast_to(an.reading_sd(par, m), (n,))
            out[key] = (mean, sd)
        return out

    def _an_start_draws(self, spec, seed):
        st = spec["start"]
        if "route" not in st:
            return None
        res = self.predict(dict(st["route"]), n_draws=4000, seed=int(seed) + 101, include_draws=True)
        return np.asarray(res["draws"]["density_rel"], float)

    def _an_is_calibration(self, o):
        cal = [(x["as_sintered_g_cm3"], x["after_900_g_cm3"]) for x in self.an_meta.get("observations", [])]
        b, a = (o.get("before") or {}).get("g_cm3"), (o.get("after") or {}).get("g_cm3")
        return any(abs(float(a) - y1) < 1e-9 and (b is None or abs(float(b) - y0) < 1e-9) for y0, y1 in cal)

    def condition_anneal(self, observations, n_nuisance=64, seed=20261023):
        """Update the anneal layer on new paired returns (importance reweighting of its stored draws, as its owner
        calibration): each observation is one body {"label", "start", "program", "second_step", "before": {"g_cm3",
        "method"} (optional), "after": {"g_cm3", "method"}, "same_piece"} with the nuisances integrated by n_nuisance
        samples per stored draw (_an_readings, the twin's reading model). Bodies are independent given the anneal
        parameters, which they share. #8's pair is already in the calibration and is refused. Returns (updated model,
        info); predict(), joint_target() and anneal_eig() of the returned model use the updated weights."""
        if not self._has_c7():
            raise ValueError("condition_anneal needs the C7 anneal layer")
        obs = [self._an_spec(dict(o), observed=True) for o in observations]
        if not obs:
            raise ValueError("condition_anneal needs at least one observation")
        for o in obs:
            if self._an_is_calibration(o):
                raise ValueError(f"observation {o.get('label')!r} is #8's pair, already in the anneal calibration "
                                 "(using it again would count it twice)")
        rng = np.random.default_rng(seed)
        lab, par, w = self._an_pool()
        ll = np.zeros(len(lab))
        for o in obs:
            sd_draws = self._an_start_draws(o, seed)
            acc = np.zeros((len(lab), n_nuisance))
            for k in range(n_nuisance):
                rd = self._an_readings(lab, par, o, rng, sd_draws)
                for key in ("before", "after"):
                    if key in rd:
                        y = float(o[key]["g_cm3"]) / RHO_TD
                        mu, sd = rd[key]
                        acc[:, k] += -0.5 * ((y - mu) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)
            ll += _lme(acc, 1)
        mechs = [m for m in self.an.ANNEAL_MECHS if m in self.an_post]
        new_dw, lz, ess = {}, {}, {}
        for m in mechs:
            pos = lab == m
            base = np.log(np.maximum(w[pos] / w[pos].sum(), 1e-300))
            tot = _lse(base + ll[pos])
            lz[m] = tot
            wv = np.exp(base + ll[pos] - tot)
            new_dw[m] = wv
            ess[m] = float(1.0 / np.sum(wv ** 2))
        zw = _lse(np.array([math.log(max(self.an_meta["mechanism_weights"][m], 1e-300)) + lz[m] for m in mechs]))
        W = {m: math.exp(math.log(max(self.an_meta["mechanism_weights"][m], 1e-300)) + lz[m] - zw) for m in mechs}
        clone = object.__new__(PhysicsModel)
        clone.__dict__.update(self.__dict__)
        clone.an_dw = new_dw
        clone.an_meta = dict(self.an_meta, mechanism_weights=W,
                             conditioned_on=list(self.an_meta.get("conditioned_on", [])) + observations)
        info = {"mechanism_weights_before": dict(self.an_meta["mechanism_weights"]), "mechanism_weights": W,
                "ess_per_mechanism": ess, "n_nuisance": n_nuisance,
                "warning": "ESS below ~200 for a mechanism that keeps weight means its stored draws no longer represent "
                           "its posterior: refit owner-side (build/c7_anneal_fit.py) with the new observation"}
        return clone, info

    def anneal_eig(self, candidates, target=None, n_outer=2000, n_inner=16000, seed=0):
        """Expected information from candidate anneal experiments (C7.1), on the anneal layer and reading model that
        predict(), the twin and condition_anneal() use. Each candidate is an _an_spec without readings' values (the
        before/after readings say which method; before may be omitted). Nested Monte Carlo: n_outer simulated outcomes
        from the posterior and n_inner posterior samples (draws with their nuisances) to evaluate each outcome's
        likelihood, so the posterior over the anneal mechanism, and over a target, follows from the same pool.
        Returns per candidate: mutual information (nats) with the mechanism label and its share of the prior entropy,
        the expected posterior mechanism probabilities by true mechanism, and, with target = {"start", "program",
        "second_step", "threshold_g_cm3": optional}, the target's annealed true density now and its expected posterior
        sd after the experiment (and P(target >= threshold) now and its expected spread after). Conditional on the
        layer's priors and measurement models; not the power of an executed protocol."""
        if not self._has_c7():
            raise ValueError("anneal_eig needs the C7 anneal layer")
        if not candidates:
            raise ValueError("anneal_eig needs at least one candidate")
        lab, par, w = self._an_pool()
        mechs = [m for m in self.an.ANNEAL_MECHS if m in self.an_post]
        prior = np.array([w[lab == m].sum() for m in mechs]) / w.sum()
        H0 = float(-np.sum(prior * np.log(np.maximum(prior, 1e-300))))
        tspec = None
        if target is not None:   # the target's true annealed density is what is scored; its reading is not used
            t = {k: v for k, v in dict(target).items() if k != "threshold_g_cm3"}
            t.setdefault("after", {"method": "archimedes"})
            tspec = self._an_spec(t, observed=False)
        out = []
        for ci, cand in enumerate(candidates):
            spec = self._an_spec(dict(cand), observed=False)
            rng = np.random.default_rng([int(seed), 97, ci])
            sd_draws = self._an_start_draws(spec, seed)
            ii = rng.choice(len(lab), size=n_inner, p=w / w.sum())
            io = rng.choice(len(lab), size=n_outer, p=w / w.sum())
            pin = {k: v[ii] for k, v in par.items()}
            pout = {k: v[io] for k, v in par.items()}
            rin = self._an_readings(lab[ii], pin, spec, rng, sd_draws)
            rout = self._an_readings(lab[io], pout, spec, rng, sd_draws)
            keys = [k for k in ("before", "after") if k in rout]
            y = {k: rout[k][0] + rout[k][1] * rng.standard_normal(n_outer) for k in keys}
            tin = None
            if tspec is not None:
                tin = self._an_readings(lab[ii], pin, tspec, rng, self._an_start_draws(tspec, seed + 1))["rho_after_true"]
            post_m = np.zeros((n_outer, len(mechs)))
            t_mean = t_var = p_thr = None
            if tin is not None:
                t_mean, t_var = np.zeros(n_outer), np.zeros(n_outer)
                thr = target.get("threshold_g_cm3")
                p_thr = np.zeros(n_outer) if thr is not None else None
            onehot = np.stack([(lab[ii] == m).astype(float) for m in mechs], 1)
            for a in range(0, n_outer, 200):
                b = min(a + 200, n_outer)
                L = np.zeros((b - a, n_inner))
                for k in keys:
                    mu, sd = rin[k]
                    L += -0.5 * ((y[k][a:b, None] - mu[None, :]) / sd[None, :]) ** 2 - np.log(sd)[None, :]
                L -= L.max(1, keepdims=True)
                E = np.exp(L)
                E /= E.sum(1, keepdims=True)
                post_m[a:b] = E @ onehot
                if tin is not None:
                    t_mean[a:b] = E @ tin
                    t_var[a:b] = E @ tin ** 2 - t_mean[a:b] ** 2
                    if p_thr is not None:
                        p_thr[a:b] = E @ (tin * RHO_TD >= float(target["threshold_g_cm3"])).astype(float)
            Hpost = -np.sum(post_m * np.log(np.maximum(post_m, 1e-300)), 1)
            mi = H0 - Hpost
            row = {"name": cand.get("name", f"candidate-{ci}"),
                   "mi_mechanism_nats": float(mi.mean()), "mi_se": float(mi.std(ddof=1) / math.sqrt(n_outer)),
                   "mechanism_entropy_nats": H0, "share_resolved": float(mi.mean() / H0) if H0 > 0 else 0.0,
                   "posterior_mechanism_by_true": {
                       m: {m2: float(post_m[lab[io] == m][:, j].mean()) for j, m2 in enumerate(mechs)}
                       for m in mechs if np.any(lab[io] == m)},
                   "readings": {k: spec[k].get("method", "unspecified") for k in keys},
                   "same_piece": spec.get("same_piece")}
            if tin is not None:
                row["target"] = {"density_g_cm3_now": {"median": float(np.median(tin)) * RHO_TD,
                                                      "sd": float(tin.std()) * RHO_TD},
                                 "expected_posterior_sd_g_cm3": float(np.sqrt(np.mean(np.maximum(t_var, 0.0)))) * RHO_TD}
                if p_thr is not None:
                    pn = float(np.mean(tin * RHO_TD >= float(target["threshold_g_cm3"])))
                    row["target"].update(threshold_g_cm3=float(target["threshold_g_cm3"]), p_above_now=pn,
                                         p_above_after_q10_q50_q90=[float(x) for x in np.quantile(p_thr, [0.1, 0.5, 0.9])])
            out.append(row)
        out.sort(key=lambda d: -d["mi_mechanism_nats"])
        return self._stamp({"ranking": out, "mechanisms": mechs, "prior_mechanism_weights": dict(zip(mechs, prior.tolist())),
                            "n_outer": n_outer, "n_inner": n_inner,
                            "method": "nested Monte Carlo on the anneal layer: outcomes simulated from the posterior "
                                      "(draws, nuisances, reading noise), each scored against an independent posterior "
                                      "pool with its own nuisances; I(M; y) = H(prior) - E[H(p(M | y))]",
                            "scope": "conditional on the anneal layer's priors, mechanisms and declared reading models; "
                                     "information about the model's mechanism label, not the power of an executable "
                                     "RPI protocol"})

    # ---------------------------------------------------------------- forward simulation for predict / p_meets_spec
    def _temperature(self, lat, hci, br, T_C, rng):
        """Declared temperature scenario (not calibrated): Kuz'min Js(T), linear HA(T), Neff = nB ln(D/dB), alpha from
        the room-temperature Hci; remanence retention R(Hci) re-evaluated at temperature."""
        if abs(T_C - 20.0) < 1e-9:
            return hci, br
        n = len(hci)
        s = rng.uniform(0.5, 0.6, n)
        cHA = rng.uniform(0.0026, 0.0032, n)
        raw, rJ = self._temperature_map(lat, hci, T_C, s, cHA)
        hci_T = np.maximum(raw, 0.0)
        R = lambda h: 1.0 / (1.0 + (lat["HR"] / np.maximum(h, 1e-3)) ** lat["kR"])
        return hci_T, br * rJ * R(hci_T) / R(hci)

    def _temperature_map(self, lat, hci, T_C, s, cHA):
        """Deterministic part of the temperature scenario: unfloored Hci(T) for a body whose room-temperature Hci is
        `hci` (alpha is derived from it, so the ratio Hci(T)/Hci(20 C) depends on the body's own Hci), and the Js ratio."""
        rJ = self.ph.kuzmin_js(T_C + 273.15, 1.0, s=s) / self.ph.kuzmin_js(293.15, 1.0, s=s)
        rH = np.clip(1.0 - cHA * (T_C - 20.0), 0.05, None)
        neff = lat["nB"] * np.log(np.maximum(lat["grain"], 0.01) / lat["dB"])
        alpha = (hci + neff * lat["Js"]) / lat["HA"]
        return alpha * lat["HA"] * rH - neff * lat["Js"] * rJ, rJ

    def _bh(self, br, hci, sq, rng):
        cal = self.meta["bhmax_shape"]
        f = cal["shape_factor"] * np.exp(cal["log_sd"] * rng.standard_normal(len(br)))
        raw = np.asarray(self.ph.bhmax(np.asarray(br), np.maximum(np.asarray(hci), 1e-3), np.asarray(sq), shape=1.0))
        return np.minimum(raw * f, np.asarray(br) ** 2 / 4.0)  # the Br^2/4 bound holds after the shape factor

    @staticmethod
    def _rho_scenario(rs):
        """Validate a declared density scenario {"mean": .., "sd": ..} (g/cm3 if mean > 1.5, else relative)."""
        if not (isinstance(rs, dict) and _is_number(rs.get("mean")) and _is_number(rs.get("sd"))
                and float(rs["mean"]) > 0 and float(rs["sd"]) >= 0):
            raise ValueError("rho_scenario must be {'mean': number > 0, 'sd': number >= 0} (g/cm3 or relative)")
        scale = RHO_TD if float(rs["mean"]) > 1.5 else 1.0
        return float(rs["mean"]) / scale, float(rs["sd"]) / scale

    def _simulate(self, route, n, seed, group="post", rho_observed=None, rho_scenario=None, sem=False, c6=False,
                  density_stage="as_sintered"):
        """C6: c6=True (needs sem=True) adds the C6 per-draw quantities to sims from their own generator, after every
        C5 draw, so the C5 arrays are unchanged.
        C7: with the anneal layer, the anneal draws come from their own generator (seed, 71); at an annealed route the
        body's density is its stage density (sims["density_rel"]; the as-sintered value is
        sims["density_rel_as_sintered"]) and C5's Br is scaled by the body's density gain over historical E's gain.
        density_stage says what rho_observed / rho_scenario describe: "as_sintered" (default; the anneal layer
        carries the body to the route's stage) or "route" (a density measured at the route's stage; not
        transformed). sims["_c7"] carries the layer's per-draw state."""
        if density_stage not in ("as_sintered", "route"):
            raise ValueError("density_stage must be 'as_sintered' or 'route'")
        r = self._route(route)
        r["_degas_expected"] = self._expected_state(r)
        rng = np.random.default_rng(seed)
        A = self._arrays(r)
        hull = self.hull(r)
        if rho_observed is not None and rho_scenario is not None:
            raise ValueError("pass either rho_observed (a measured body) or rho_scenario (an assumed density), not both")
        if rho_observed is not None:
            rho_obs = float(rho_observed)
            rho_obs = rho_obs / RHO_TD if rho_obs > 1.5 else rho_obs
            rho = np.full(n, rho_obs) + 0.005 * rng.standard_normal(n)
            dlab = np.array(["observed"] * n)
        elif rho_scenario is not None:
            mu, sd = self._rho_scenario(rho_scenario)
            rho = np.clip(mu + sd * rng.standard_normal(n), 0.3, 1.02)
            dlab = np.array(["scenario"] * n)
        else:
            dlab, didx = self._assign_bma("density", n, rng, group)
            mu, sig = self._density_mean(A, r, dlab, didx, rng.standard_normal((n, 1)), group)
            rho = np.clip(mu + sig * rng.standard_t(5.0, n), 0.3, 1.02)
        mlab, midx = self._assign_bma("magnetic", n, rng, group)
        ind = {k: self._assign_indicator(k, mlab, midx, np.random.default_rng([int(seed), 47]), group)
               for k in self._indicators()}
        degas_ind, eps = self._degas_state(r, rng.random(n), np.random.default_rng([int(seed), 31]))
        z_hull = rng.standard_normal(n)
        sig_spec = np.array([self.post[m]["sig_spec"][i] if group == "post" else self.prior[m]["sig_spec"][i]
                             for m, i in zip(mlab, midx)])
        u = sig_spec * rng.standard_normal(n)  # a new physical body: its own specimen effect (before the Hci cap)
        lat = self._magnetic_latent(A, r, mlab, midx, rho, degas_ind, z_hull, u, group, eps=eps, indicators=ind,
                                    sem=sem)
        c7 = (self._c7_state(r, rho, lat if sem else None, mlab, midx, group, np.random.default_rng([int(seed), 71]),
                             density_stage) if self._has_c7() else None)
        if c7 is not None:   # C7.1: H4's grain term follows the hold (no random draw)
            lat["lh"] = self._c7_apply_lh_shift(lat["lh"], c7["h4_shift"])
        hci_meas = np.exp(lat["lh"]) * np.exp(lat["sig_piece"] * rng.standard_normal(n))
        br = np.exp(lat["lbr"] + lat["sig_Br"] * rng.standard_normal(n))
        if c7 is not None and c7["br_factor"] is not None:   # C7: no random draw, so every C5 draw below is unchanged
            br = br * c7["br_factor"]
        sq = 1.0 / (1.0 + np.exp(-(lat["sq"] + lat["sig_sq"] * rng.standard_normal(n))))
        hci_T, br_T = self._temperature(lat, hci_meas, br, float(r["T_measure_C"]), rng)
        bh = self._bh(br_T, hci_T, sq, rng)
        grain = lat["grain"]
        if c7 is not None and "grain_stage" in c7 and c7["route_stage"] != "as_sintered":
            grain = c7["grain_stage"][c7["route_stage"]]
        rho_route = c7["stage_rho"][c7["route_stage"]] if c7 is not None else rho
        sims = dict(density_rel=rho_route, grain_um=grain, Hci_kOe=hci_T, Br_kGs=br_T, Hk_over_Hci=sq, BHmax_MGOe=bh)
        if c7 is not None:
            sims["density_rel_as_sintered"] = np.asarray(rho, float)
            sims["_c7"] = c7
        if sem:  # C5: own generator, so every draw above is unchanged
            sims.update(self._sem_sims(lat, np.random.default_rng([int(seed), 59])))
        if c6:  # C6: own generator, after every C5 draw
            if not sem:
                raise ValueError("the C6 outputs need the C5 SEM/BSE latents (sem=True)")
            sims.update(self._c6_sims(r, lat, mlab, midx, rho, group, np.random.default_rng([int(seed), 61]), c7=c7))
        hull_m = self.hull(r, rho=float(np.median(rho)))["magnetic"]
        r.pop("_degas_expected", None)
        return r, sims, dlab, mlab, dict(density=hull["density"], magnetic=hull_m), ind

    def _has_sem(self):
        """True when every coercivity mechanism carries the C5 SEM/BSE draws (posterior and prior)."""
        return bool(self.post) and all(
            all(k in src[m] for k in C5_SEM_PARAMS) and ("Tw" in src[m] or "c5_Tw" in src[m])
            for src in (self.post, self.prior) for m in self.w["magnetic"] if m in src)

    def _sem_sims(self, lat, rng):
        """C5 measured SEM/BSE quantities of one new body per draw: G_stage = D_stage x c_m x exp(e_stage), e_stage ~
        N(0, GRAIN_FIELD_LOG_SD^2) independent per stage (field sampling), c_m the draw's method factor shared by both
        stages of the piece; the paired change G_after_900_520 - G_as_sintered and ratio from the same draw and piece;
        the probability of a 'spread' BSE reading Phi((W_stage - W*) / tau) at each stage, W_after = 1 - (1 - W)(1 - r)."""
        n = len(lat["W_bse"])
        sd = float(self.fw.GRAIN_FIELD_LOG_SD)
        g = {s: lat["g_" + s] * lat["c5_cm"] * np.exp(sd * rng.standard_normal(n)) for s in SEM_STAGES}
        out = {"grain_size_sem_um_as_sintered": g["as_sintered"],
               "grain_size_sem_um_after_900_520": g["after_900_520"],
               "anneal_grain_change_um": g["after_900_520"] - g["as_sintered"],
               "anneal_grain_change_ratio": g["after_900_520"] / g["as_sintered"]}
        for s in SEM_STAGES:
            W = self.fw.bse_stage_continuity({"c5_rann": lat["c5_rann"]}, lat["W_bse"], s, np)
            out["gb_phase_bse_W_" + s] = W
            out["gb_phase_bse_p_spread_" + s] = _std_normal_cdf((W - lat["c5_Wstar"]) / lat["c5_tau"])
        return out

    def _c5_status(self, r):
        """Per-output C5 status (code and note) from the package metadata; a declared feedstock reads declared."""
        st = dict(self.meta.get("c5", {}).get("status", {}))
        out = {}
        for k in SEM_OUTPUTS + ("gb_phase_bse_after_900_520",):
            s = dict(st.get(k, {"code": "declared_uncalibrated", "note": "no status in this package's metadata"}))
            if self._feedstock_declared(r):
                s = {"code": STATUS_DECLARED, "note": "declared feedstock particle size: the grain hook, the oxide term "
                     "and the feedstock layer are uncalibrated priors; " + s.get("note", "")}
            out[k] = s
        return out

    def _sem_summary(self, x, lab, ind, prior_x, out_of_hull, log_scale=True):
        """Summary of one continuous C5 output in the shape of the existing outputs (quantiles, flags, splits)."""
        x = np.asarray(x, float)
        tr = (lambda v: np.log(np.maximum(v, 1e-6))) if log_scale else (lambda v: v)
        groups = {m: x[lab == m] for m in np.unique(lab)}
        tx, tp = tr(x), tr(np.asarray(prior_x, float))
        v_post = float(np.var(tx))
        between = float(np.var(np.concatenate([np.full(len(g), np.mean(tr(g))) for g in groups.values()]))) \
            if len(groups) > 1 else 0.0
        share = between / v_post if v_post > 0 else 0.0
        ratio = (_iqr(tx) / _iqr(tp)) ** 2 if _iqr(tp) > 0 else 1.0
        extra = {}
        if "load_timing" in ind:
            lt = ind["load_timing"]
            extra["by_load_timing"] = {L: dict(median=float(np.median(x[lt == L])), q05=float(np.quantile(x[lt == L], 0.05)),
                                               q95=float(np.quantile(x[lt == L], 0.95)), n=int((lt == L).sum()))
                                       for L in LT_LEVELS if (lt == L).any()}
        return dict(mean=float(np.mean(x)), median=float(np.median(x)), sd=float(np.std(x)),
                    q05=float(np.quantile(x, 0.05)), q25=float(np.quantile(x, 0.25)), q75=float(np.quantile(x, 0.75)),
                    q95=float(np.quantile(x, 0.95)), n=int(len(x)),
                    by_mechanism={m: dict(median=float(np.median(g)), q05=float(np.quantile(g, 0.05)),
                                          q95=float(np.quantile(g, 0.95)), n=int(len(g))) for m, g in groups.items()},
                    posterior_to_prior_spread_ratio=ratio, between_mechanism_share=share,
                    prior_dominated=bool(ratio > 0.5 or out_of_hull), mechanism_dominated=bool(share > 0.5),
                    **extra, **_distribution(x))

    def _sem_outputs(self, r, sims, mlab, ind, prs, hull):
        """predict()'s C5 outputs: grain_size_sem_um and gb_phase_bse at both stages; anneal_grain_change (paired)."""
        st = self._c5_status(r)
        oh = hull["magnetic"]["out_of_hull"]
        g = {s: self._sem_summary(sims["grain_size_sem_um_" + s], mlab, ind, prs["grain_size_sem_um_" + s], oh)
             for s in SEM_STAGES}
        out = {"grain_size_sem_um": dict(g, units="um", status=st["grain_size_sem_um"]["code"],
                                         status_note=st["grain_size_sem_um"]["note"],
                                         measurement_model="ln G = ln D_stage + ln c_m + e: D_stage the model grain "
                                         "(grain-size law, feedstock hook included), c_m the method factor (per draw, "
                                         "log-normal median 1, log-sd 0.2, declared; shared by both stages of a piece), "
                                         "e field sampling (log-sd 0.15, declared, independent per stage)",
                                         stage_note="both stages of the route's body regardless of route['anneal']: "
                                         "as sintered, and after 900 C/2 h + 520 C/2 h (the 520 C step adds no growth "
                                         "in the grain law); grain_um is unchanged")}
        out["anneal_grain_change"] = {
            "delta_um": self._sem_summary(sims["anneal_grain_change_um"], mlab, ind, prs["anneal_grain_change_um"], oh,
                                          log_scale=False),
            "ratio": self._sem_summary(sims["anneal_grain_change_ratio"], mlab, ind, prs["anneal_grain_change_ratio"], oh),
            "status": st["anneal_grain_change"]["code"] if self._feedstock_declared(r) else STATUS_MODELLED_PAIR,
            "status_note": ("modelled pair: both stages from the same draw and the same piece (c_m shared, so it "
                            "cancels in the ratio; field sampling independent per stage); " + st["anneal_grain_change"]["note"]),
            "units": {"delta_um": "um (after 900+520 minus as sintered)", "ratio": "after / as sintered"}}
        lt = ind.get("load_timing")
        bse = {}
        for s, sk in (("as_sintered", "gb_phase_bse"), ("after_900_520", "gb_phase_bse_after_900_520")):
            ps, W = np.asarray(sims["gb_phase_bse_p_spread_" + s]), np.asarray(sims["gb_phase_bse_W_" + s])
            bse[s] = {
                "p_spread": float(ps.mean()), "p_pooled": float(1.0 - ps.mean()), "n": int(len(ps)),
                "prior_predictive_p_spread": float(np.mean(prs["gb_phase_bse_p_spread_" + s])),
                "W": dict(median=float(np.median(W)), q05=float(np.quantile(W, 0.05)), q95=float(np.quantile(W, 0.95)),
                          quantiles={k: float(v) for k, v in zip(QUANTILE_KEYS, np.quantile(W, QUANTILE_LEVELS))}),
                "by_mechanism": {m: float(ps[mlab == m].mean()) for m in np.unique(mlab)},
                "by_load_timing": ({L: float(ps[lt == L].mean()) for L in LT_LEVELS if (lt == L).any()}
                                   if lt is not None else None),
                "status": st[sk]["code"], "status_note": st[sk]["note"]}
        out["gb_phase_bse"] = dict(
            bse, status=st["gb_phase_bse"]["code"], status_note=st["gb_phase_bse"]["note"],
            observation_model="P(spread) = Phi((W_stage - W*) / tau); W = sigmoid((T_peak - Tw)/ww) x clip(1 - zeta "
                              "Pi_LT(route), 0, 1) x exp(-kappa x oxidation penalty) as sintered (the oxide term only "
                              "for a declared feedstock); W_after_900_520 = 1 - (1 - W)(1 - r); W*, tau, zeta, kappa, r "
                              "declared; Tw, ww fitted in H2, prior draws elsewhere",
            classes=["spread", "pooled"],
            not_modelled="EDS oxygen or oxide inclusion fractions, the Nd-rich phase area fraction, porosity seen in "
                         "SEM, and grain-size distributions beyond the field log-sd")
        return out

    # ---------------------------------------------------------------- C6 microstructure layer
    def _has_c6(self):
        """True when microstructure.py is loaded and every coercivity mechanism carries the c6_* draws (posterior and
        prior) next to its C5 SEM/BSE draws."""
        if self.ms is None or not self._has_sem():
            return False
        mechs = [m for m in self.w["magnetic"] if m in self.post]
        return bool(mechs) and all(m in src and all(k in src[m] for k in self.ms.C6_PARAMS)
                                   for src in (self.c6post, self.c6prior) for m in mechs)

    def _c6_params(self, mlab, midx, group):
        """The C6 per-draw parameters of the draws (mlab, midx), as (n,) arrays."""
        src = self.c6post if group == "post" else self.c6prior
        out = {k: np.empty(len(mlab)) for k in self.ms.C6_PARAMS}
        for m in np.unique(mlab):
            pos = np.where(mlab == m)[0]
            for k in out:
                out[k][pos] = src[m][k][midx[pos]]
        return out

    def _c6_oxide_factor(self, r, lat, mlab, midx, group):
        """A3 amount term: the C5 oxide term exp(-c5_kox x the C4 oxidation penalty) for a declared feedstock; None
        (neutral) at the fitted 4 um feedstock, as in C5."""
        if not self._feedstock_declared(r):
            return None
        src = self.post if group == "post" else self.prior
        fd = float(r[FEEDSTOCK_KEY])
        ox = np.empty(len(mlab))
        for m in np.unique(mlab):
            pos = np.where(mlab == m)[0]
            ox[pos] = self.fw.feedstock_oxidation_penalty(self._params(src, m, midx[pos]), fd, np)[:, 0]
        return np.exp(-lat["c5_kox"] * ox)

    def _c6_sims(self, r, lat, mlab, midx, rho, group, rng, c7=None):
        """C6 quantities of one new body per draw (microstructure.py): per stage the SEM-protocol grain distribution
        (median with field sampling, d90) and the fracture-SE intercept (field sampling independent per stage, the
        method factor shared, so it cancels in the anneal ratio); the SEM porosity reading; the A3 latents and readings
        at every stage from the same draw (paired anneal); and the C5 latents the coercivity link reads."""
        ms = self.ms
        P = self._c6_params(mlab, midx, group)
        n = len(mlab)
        rho = np.asarray(rho, float)
        D = {"as_sintered": lat["g_as_sintered"], "after_900": lat["g_after_900_520"],
             "after_900_520": lat["g_after_900_520"]}  # the C5 grain law gives the 520 C step no growth
        rho_s = {s: rho for s in C6_STAGES}
        if c7 is not None:   # C7: the C5 grain after the anneal's hold, and each stage's own density
            D = dict(c7.get("grain_stage", D))
            rho_s = {s: np.asarray(c7["stage_rho"][s], float) for s in C6_STAGES}
        rho_sec = rho
        if c7 is not None:   # C7.1: a section is read at the body's route stage (as in the twin)
            rho_sec = np.asarray(c7["stage_rho"][c7["route_stage"]], float)
        out = {"c6_W": np.array(lat["W_bse"], float), "c6_grain_sigma_ln": P["c6_sig"],
               "c6_planar": ms.planar(rho_sec).astype(float), "c6_rho": np.array(rho_sec, float)}
        zf = rng.standard_normal((len(C6_STAGES), n))
        zg = rng.standard_normal((len(C6_STAGES), n))
        for i, s in enumerate(C6_STAGES):
            out["c6_grain_model_um_" + s] = np.array(D[s], float)
            out["c6_fracture_intercept_um_" + s] = ms.fracture_latent(P, D[s], rho_s[s]) * np.exp(P["c6_sfield"] * zf[i])
            out["c6_rho_stage_" + s] = np.asarray(rho_s[s], float)
            med = ms.grain_median(P, D[s]) * np.exp(P["c6_sfield"] * zg[i])
            out["c6_grain_median_um_" + s] = med
            out["c6_grain_d90_um_" + s] = ms.d90(med, P["c6_sig"])
        out["c6_porosity_area_fraction_sem"] = np.minimum(ms.porosity_latent(rho_sec) * np.exp(
            P["c6_pbias"] + ms.POROSITY_LOG_SD * rng.standard_normal(n)), 1.0)
        ox = self._c6_oxide_factor(r, lat, mlab, midx, group)
        for s in C6_STAGES:
            la = ms.a3_latent(P, lat["W_bse"], s, ox)
            for k in ms.A3_KEYS:
                out[f"c6_{k}_latent_{s}"] = la[k]
                out[f"c6_{k}_{s}"] = ms.a3_measure(k, la[k], rng.standard_normal(n))
        for k in ("lh", "Js", "HA", "nB", "dB", "HR", "kR", "sig_spec", "sig_piece"):  # read by the coercivity link
            out["c6_c5_" + k] = np.array(lat[k], float)
        out["c6_ln_chem_offset"] = np.array(lat["ln_chem_offset"], float)   # link v2 chemistry coupling
        return out

    def _c6_status(self, r):
        """Per-output C6 status (code and note) from the package metadata; a declared feedstock reads declared."""
        st = dict(self.meta.get("c6", {}).get("status", {}))
        out = {}
        for k in C6_OUTPUTS:
            s = dict(st.get(k, {"code": "declared_uncalibrated", "note": "no status in this package's metadata"}))
            if self._feedstock_declared(r) and k not in ("sps_log", "coercivity_link"):
                s = {"code": STATUS_DECLARED, "note": "declared feedstock particle size: the C5 grain hook, the oxide "
                     "term and the feedstock layer are uncalibrated priors; " + s.get("note", "")}
            out[k] = s
        return out

    def _c6_summary(self, x, lab, ind, prior_x, out_of_hull, log_scale=True, mask=None, prior_mask=None):
        """_sem_summary over the draws in mask (planar sections); None when fewer than 10 draws remain."""
        x, px = np.asarray(x, float), np.asarray(prior_x, float)
        if mask is not None:
            x, lab = x[mask], np.asarray(lab)[mask]
            ind = {k: np.asarray(v)[mask] for k, v in ind.items()}
        if prior_mask is not None and int(np.sum(prior_mask)) >= 10:  # else the whole prior predictive is the reference
            px = px[prior_mask]
        if len(x) < 10 or len(px) < 10:
            return None
        return self._sem_summary(x, lab, ind, px, out_of_hull, log_scale=log_scale)

    @staticmethod
    def _check_threshold(coarse_threshold_nm):
        if not (_is_number(coarse_threshold_nm) and float(coarse_threshold_nm) > 0):
            raise ValueError("coarse_threshold_nm must be a positive number (nm): a coarse number fraction needs its "
                             "threshold")
        return float(coarse_threshold_nm)

    def _c6_outputs(self, r, sims, mlab, ind, prs, hull, coarse_threshold_nm=None):
        """predict()'s C6 outputs (A1-A4 of C6_SPEC): summaries over draws, per stage where the quantity has stages."""
        ms = self.ms
        st = self._c6_status(r)
        oh = hull["magnetic"]["out_of_hull"]
        out = {}
        g = {}
        for s in C6_STAGES:
            med, sig = sims["c6_grain_median_um_" + s], sims["c6_grain_sigma_ln"]
            e = {"median_um": self._c6_summary(med, mlab, ind, prs["c6_grain_median_um_" + s], oh),
                 "d90_um": self._c6_summary(sims["c6_grain_d90_um_" + s], mlab, ind, prs["c6_grain_d90_um_" + s], oh),
                 "model_grain_um": self._c6_summary(sims["c6_grain_model_um_" + s], mlab, ind,
                                                    prs["c6_grain_model_um_" + s], oh)}
            if coarse_threshold_nm is None:
                e["coarse_fraction"] = {"status": "requires_threshold",
                                        "note": "pass coarse_threshold_nm (nm): a coarse number fraction is defined "
                                                "only for a stated threshold"}
            else:
                thr = float(coarse_threshold_nm) / 1000.0
                cf = ms.coarse_fraction(med, sig, thr)
                pcf = ms.coarse_fraction(prs["c6_grain_median_um_" + s], prs["c6_grain_sigma_ln"], thr)
                e["coarse_fraction"] = dict(self._c6_summary(cf, mlab, ind, pcf, oh, log_scale=False),
                                            threshold_nm=float(coarse_threshold_nm),
                                            units="number fraction of grains larger than the threshold")
            g[s] = e
        sg = np.asarray(sims["c6_grain_sigma_ln"], float)
        out["grain_sem_dist"] = dict(
            g, units="um", sigma_ln=dict(median=float(np.median(sg)), q05=float(np.quantile(sg, 0.05)),
                                         q95=float(np.quantile(sg, 0.95))),
            coarse_threshold_nm=None if coarse_threshold_nm is None else float(coarse_threshold_nm),
            status=st["grain_sem_dist"]["code"], status_note=st["grain_sem_dist"]["note"],
            measurement_model="log-normal grain distribution an SEM grain-size protocol would report: mean = D_stage x "
                              "c_m6 (D_stage the C5 model grain, c_m6 the method factor, log-normal median 1, log-sd "
                              "0.47 before calibration, one per world), sigma_ln per draw (prior from the metrology "
                              "log-normal fits, 0.46-0.51); median = mean x exp(-sigma^2/2) with field sampling "
                              "(log-sd c6_sfield); d90 = median x exp(1.2816 sigma); coarse number fraction above t = "
                              "Phi((ln median - ln t) / sigma)",
            stage_note="as sintered, after 900 C/2 h and after 900 C/2 h + 520 C/2 h of the same piece regardless of "
                       "route['anneal']; the C5 grain law gives the 520 C step no growth; model_grain_um is the C5 "
                       "model grain (no method factor)")
        fr = {s: self._c6_summary(sims["c6_fracture_intercept_um_" + s], mlab, ind,
                                  prs["c6_fracture_intercept_um_" + s], oh) for s in C6_STAGES}
        base, pbase = sims["c6_fracture_intercept_um_as_sintered"], prs["c6_fracture_intercept_um_as_sintered"]
        ratio = {s: self._c6_summary(sims["c6_fracture_intercept_um_" + s] / base, mlab, ind,
                                     prs["c6_fracture_intercept_um_" + s] / pbase, oh)
                 for s in ("after_900", "after_900_520")}
        rho = np.asarray(sims["c6_rho"], float)
        out["fracture_intercept_um"] = dict(
            fr, anneal_ratio=dict(ratio, note="same draw and piece: c_m6 and the feature factor cancel; field sampling "
                                              "is independent per stage"),
            units="um", porous_body_share=float(np.mean(rho < ms.RHO_DENSE_FRACTURE)),
            status=st["fracture_intercept_um"]["code"], status_note=st["fracture_intercept_um"]["note"],
            measurement_model="mean linear intercept of a fracture SE image (frozen-script protocol, 41 um fields): "
                              "D_stage x 2/3 x phi(rho) x c_m6 x exp(e); 2/3 sphere stereology; phi = the declared "
                              "feature factor below ~95 %TD (particles and ligaments, not grains), 1 for dense bodies; "
                              "e field sampling, log-sd c6_sfield per draw, independent per stage",
            stage_note="as for grain_sem_dist; C5's grain_size_sem_um and anneal_grain_change are unchanged")
        pl, ppl = np.asarray(sims["c6_planar"]) > 0.5, np.asarray(prs["c6_planar"]) > 0.5
        p_planar = float(pl.mean())
        flat = p_planar >= 0.5
        sect = ("planar (at or above 92 %TD, the least dense planar section on record)"
                if float(np.median(rho)) >= ms.RHO_PLANAR_OBSERVED else
                "planar by declaration (80-92 %TD: no section on record)" if flat else
                f"not planar below {ms.RHO_PLANAR:.2f} relative density")
        nfl = (f"status not_planar: in most draws the body is below {ms.RHO_PLANAR:.2f} relative density, where a "
               "polished section of a porous compact is not planar (relief and particle edges, no matrix mode); no "
               "value")

        notpl = None if flat else dict(value=None, p_planar=p_planar, status="not_planar", status_note=nfl,
                                       section=sect)
        out["porosity_area_fraction_sem"] = notpl or dict(
            value=self._c6_summary(sims["c6_porosity_area_fraction_sem"], mlab, ind,
                                   prs["c6_porosity_area_fraction_sem"], oh, log_scale=False, mask=pl, prior_mask=ppl),
            p_planar=p_planar, section=sect, status=st["porosity_area_fraction_sem"]["code"],
            status_note=st["porosity_area_fraction_sem"]["note"],
            measurement_model="BSE dark area fraction of a planar section: (1 - rho) x exp(b + e), b the per-world "
                              "method bias, e ~ N(0, 0.5^2) per reading (the dark fraction brackets rho-porosity only "
                              "within a factor of 3-4); over the draws with a planar section; density does not change "
                              "with the anneal in C5")
        for k in ms.A3_KEYS:
            if notpl is not None:
                out[k] = dict(notpl, **{s: None for s in C6_STAGES})
                continue
            logk = k == "pocket_ecd_um"
            out[k] = dict({s: self._c6_summary(sims[f"c6_{k}_{s}"], mlab, ind, prs[f"c6_{k}_{s}"], oh,
                                               log_scale=logk, mask=pl, prior_mask=ppl) for s in C6_STAGES},
                          p_planar=p_planar, section=sect, status=st[k]["code"], status_note=st[k]["note"],
                          units="um" if logk else "fraction",
                          basis=self._A3_BASIS[k])
        out["nd_rich_area_fraction"]["observation_model"] = (  # shared by the five A3 outputs
            "latents per draw from the C5 continuity W (as sintered) and a declared amount term: area fraction = "
            "v_Nd x exp(b_f (W - 0.5)) x the C5 oxide term (declared feedstock only); pocket ECD = e0 x exp(b_e (W - "
            "0.5)); pocket, elongated and faint-line shares = sigmoid(a + b (W - 0.5)); after 900 C/2 h + 520 C/2 h each "
            "shifts by its declared paired change (after 900 C by the share lam900 of it); readings carry declared "
            "body-and-field scatter. SEM at 13-27 nm/px cannot see the 2-3 nm boundary film")
        out["sps_log"] = {"status": ms.SPS_STATUS if r["pressure_mode"] == "cooling"
                          else ms.SPS_STATUS + "; " + ms.SPS_STATUS_DECLARED_TIMING,
                          "note": "call sps_log(route, n_draws=..., seed=...) for the trace; its end state equals this "
                                  "call's density_rel draws for the same route, seed and n_draws"}
        return out

    _A3_BASIS = {"nd_rich_area_fraction": "bright (Nd-rich) area fraction of a planar polished BSE section, Otsu "
                                          "threshold, 26.98 nm/px",
                 "pocket_ecd_um": "area-weighted equivalent-circle diameter of the bright pockets (+27.5 grey-level "
                                  "offset basis)",
                 "pocket_share": "share of the bright area in pockets thicker than 0.20 um (thickness split)",
                 "elongated_share": "share of the bright area in objects of aspect ratio >= 3",
                 "faint_line_coverage": "faint bright boundary lines as a share of the expected boundary length (T10 "
                                        "lines, 2.5 um intercept basis); readings at or below the detection floor "
                                        "(about 0.3 %) are noise"}

    def _link_state(self, r, sims, mlab, ind):
        """The per-draw state passed to coercivity_link.link(state, rng) (C6_SPEC A5; keys LINK_STATE_KEYS):
        n; route (resolved); alignment_cos_theta and alignment_declared (A6); stages; rho_rel (the body's relative
        density draw); grain_model_um {stage: C5 model grain}; grain_sigma_ln; grain_sem_median_um and
        fracture_intercept_um {stage: SEM readings}; nd_rich {stage: {A3 key: latent}}; W (C5 continuity, as sintered);
        c5 {ln_hci (route-level ln Hci with the body's specimen effect), Hci_kOe, Br_kGs, Hk_over_Hci, BHmax_MGOe
        (the C5 measured outputs), Js, HA, nB, dB, HR, kR, sig_spec, sig_piece (the draw's C5 defect, thermal and
        scatter terms), mechanism, load_timing}; J_GB_posterior_T (never drawn). Arrays are (n,) and aligned."""
        al = self._alignment(r)
        c5 = {k: np.asarray(sims["c6_c5_" + k], float) for k in ("Js", "HA", "nB", "dB", "HR", "kR", "sig_spec",
                                                                 "sig_piece")}
        c5.update(ln_hci=np.asarray(sims["c6_c5_lh"], float), mechanism=np.asarray(mlab).astype(str),
                  load_timing=(np.asarray(ind["load_timing"]).astype(str) if "load_timing" in ind else None),
                  **{k: np.asarray(sims[k], float) for k in ("Hci_kOe", "Br_kGs", "Hk_over_Hci", "BHmax_MGOe")})
        return {"n": int(len(mlab)), "route": {k: v for k, v in r.items() if not str(k).startswith("_")},
                "alignment_cos_theta": al[ALIGN_KEY], "alignment_declared": al["declared_extrapolation"],
                "stages": C6_STAGES, "rho_rel": np.asarray(sims["density_rel"], float),
                "grain_model_um": {s: sims["c6_grain_model_um_" + s] for s in C6_STAGES},
                "grain_sigma_ln": np.asarray(sims["c6_grain_sigma_ln"], float),
                "grain_sem_median_um": {s: sims["c6_grain_median_um_" + s] for s in C6_STAGES},
                "fracture_intercept_um": {s: sims["c6_fracture_intercept_um_" + s] for s in C6_STAGES},
                "nd_rich": {s: {k: sims[f"c6_{k}_latent_{s}"] for k in self.ms.A3_KEYS} for s in C6_STAGES},
                "W": np.asarray(sims["c6_W"], float), "c5": c5, "J_GB_posterior_T": dict(J_GB_POSTERIOR_T),
                "ln_hci_chem_offset": np.asarray(sims["c6_ln_chem_offset"], float),
                "rho_rel_by_stage": ({s: np.asarray(sims["c6_rho_stage_" + s], float) for s in C6_STAGES}
                                     if "c6_rho_stage_as_sintered" in sims else None)}

    def _link_outputs(self, r, sims, mlab, ind, seed):
        """A5 integration point: coercivity_link.link(state, rng) when the module is loaded (its own generator
        (seed, 67)); otherwise the link outputs are omitted with status LINK_STATUS_MISSING."""
        if self.link is None:
            return {"status": LINK_STATUS_MISSING,
                    "note": "the coercivity link (C6 work package 1) is not part of this load; hci_link_kOe, "
                            "br_link_kGs, hk_over_hci_link and bhmax_link_MGOe are omitted. The C5 magnetic outputs "
                            "(Hci_kOe and so on) are unchanged"}
        out = self.link.link(self._link_state(r, sims, mlab, ind), np.random.default_rng([int(seed), 67]))
        if not isinstance(out, dict):
            raise TypeError("coercivity_link.link must return a dict")
        return dict(out, module_sha256=self.hashes.get("coercivity_link"))

    def _c6_block(self, r):
        c6m = self.meta.get("c6", {})
        return {"alignment": self._alignment(r), "stages": list(C6_STAGES),
                "calibration_stage": (c6m.get("calibration") or {}).get("stage", "declared_uncalibrated"),
                "data_cutoff_step": c6m.get("data_cutoff_step"),
                "resolution_limit": "SEM at 13-27 nm/px cannot see the 2-3 nm boundary film: the Nd-rich outputs "
                                    "bound the thick (resolvable) phase only",
                "readings_note": "the SEM outputs simulate readings of RPI-style images by the owner's frozen metrology "
                                 "protocol; they are model outputs, not measurements"}

    # ---------------------------------------------------------------- public API
    def predict(self, route, n_draws=4000, seed=0, rho_observed=None, include_draws=False, rho_scenario=None,
                coarse_threshold_nm=None, density_stage="as_sintered"):
        """Posterior predictive of a *measured* as-processed body (specimen-to-specimen and measurement scatter
        included). rho_observed (g/cm3 or relative): condition the magnetics on a measured density (an existing body).
        rho_scenario {"mean", "sd"}: an assumed density with its uncertainty, for a body not yet made.
        C5: with a posterior that declares them, outputs also carry grain_size_sem_um, anneal_grain_change and
        gb_phase_bse (README, "SEM and BSE outputs"); every other output is computed exactly as without them.
        C6: with the microstructure layer, outputs also carry grain_sem_dist (coarse number fraction only with
        coarse_threshold_nm), fracture_intercept_um, porosity_area_fraction_sem, the five Nd-rich phase outputs, an
        sps_log pointer and coercivity_link; every C5 output is computed exactly as in C5.
        C7: at an annealed route density_rel is the body's density at that stage (the anneal layer), with
        density_rel_as_sintered beside it; res["c7"] reports the stage densities by anneal mechanism and the readings a
        density instrument would return; res["route_resolution"] reports aliases, alignment and the anneal hold.
        density_stage: what rho_observed / rho_scenario describe ("as_sintered", default, or "route")."""
        sem = self._has_sem()
        c6 = self._has_c6()
        if coarse_threshold_nm is not None:
            if not c6:
                raise ValueError("coarse_threshold_nm needs a package with the C6 microstructure layer")
            coarse_threshold_nm = self._check_threshold(coarse_threshold_nm)
        r, sims, dlab, mlab, hull, ind = self._simulate(route, n_draws, seed, "post", rho_observed, rho_scenario, sem,
                                                        c6, density_stage=density_stage)
        _, prs, _, _, _, _ = self._simulate(route, max(1000, n_draws // 2), seed + 1, "prior", rho_observed,
                                            rho_scenario, sem, c6, density_stage=density_stage)
        resolution = r.pop("_resolution", None)
        res = {"route": r, "route_resolution": resolution, "mechanism_weights": self.w, "out_of_hull": hull,
               "outputs": {}}
        for k in OUTPUTS:
            x = np.asarray(sims[k], float)
            logk = k not in ("density_rel", "Hk_over_Hci")
            tr = (lambda v: np.log(np.maximum(v, 1e-6))) if logk else (lambda v: v)
            lab = dlab if k == "density_rel" else mlab
            groups = {m: x[lab == m] for m in np.unique(lab)}
            tx, tp = tr(x), tr(np.asarray(prs[k], float))
            v_post = float(np.var(tx))
            between = float(np.var(np.concatenate([np.full(len(g), np.mean(tr(g))) for g in groups.values()]))) \
                if len(groups) > 1 else 0.0
            share = between / v_post if v_post > 0 else 0.0
            ratio = (_iqr(tx) / _iqr(tp)) ** 2 if _iqr(tp) > 0 else 1.0
            module = "density" if k == "density_rel" else "magnetic"
            observed = k == "density_rel" and rho_observed is not None
            extra = {}
            if module == "magnetic" and "load_timing" in ind:
                lt = ind["load_timing"]
                extra["by_load_timing"] = {L: dict(median=float(np.median(x[lt == L])),
                                                   q05=float(np.quantile(x[lt == L], 0.05)),
                                                   q95=float(np.quantile(x[lt == L], 0.95)), n=int((lt == L).sum()))
                                           for L in LT_LEVELS if (lt == L).any()}
            res["outputs"][k] = dict(
                mean=float(np.mean(x)), median=float(np.median(x)), sd=float(np.std(x)),
                q05=float(np.quantile(x, 0.05)), q25=float(np.quantile(x, 0.25)), q75=float(np.quantile(x, 0.75)),
                q95=float(np.quantile(x, 0.95)),
                by_mechanism={m: dict(median=float(np.median(g)), q05=float(np.quantile(g, 0.05)),
                                      q95=float(np.quantile(g, 0.95)), n=int(len(g))) for m, g in groups.items()},
                posterior_to_prior_spread_ratio=ratio, between_mechanism_share=share,
                prior_dominated=bool(False if observed else (ratio > 0.5 or hull[module]["out_of_hull"])),
                mechanism_dominated=bool(share > 0.5), conditioned_on_observed_density=observed,
                density_scenario_declared=bool(rho_scenario is not None), **extra, **_distribution(x),
                status=self._feedstock_info(r)["status"])
        d = res["outputs"]["density_rel"]
        res["outputs"]["density_g_cm3"] = {k: d[k] * RHO_TD for k in ("mean", "median", "sd", "q05", "q25", "q75", "q95")}
        res["outputs"]["density_g_cm3"]["quantiles"] = {k: v * RHO_TD for k, v in d["quantiles"].items()}
        if "density_rel_as_sintered" in sims:
            x = np.asarray(sims["density_rel_as_sintered"], float)
            res["outputs"]["density_rel_as_sintered"] = dict(
                median=float(np.median(x)), q05=float(np.quantile(x, 0.05)), q95=float(np.quantile(x, 0.95)),
                quantiles={k: float(v) for k, v in zip(QUANTILE_KEYS, np.quantile(x, QUANTILE_LEVELS))},
                note="the same body before the anneal (C7); density_rel is its density at the route's stage")
        if sem:
            res["outputs"].update(self._sem_outputs(r, sims, mlab, ind, prs, hull))
        if c6:
            res["outputs"].update(self._c6_outputs(r, sims, mlab, ind, prs, hull, coarse_threshold_nm))
            res["outputs"]["coercivity_link"] = self._link_outputs(r, sims, mlab, ind, seed)
            res["c6"] = self._c6_block(r)
        res["c7"] = self._c7_block(r, sims.get("_c7"), seed)
        js = 16.1
        br_iso = js * d["median"] * float(r["alignment_cos"])
        res["ceilings"] = {"Br_ideal_kGs_Js_rho_cos": br_iso,
                           "BHmax_bound_MGOe_Br2_over_4": br_iso ** 2 / 4.0,
                           "isotropic_full_density_BHmax_MGOe": (js * 0.5) ** 2 / 4.0,
                           "Br_needed_for_45_MGOe_kGs": math.sqrt(4 * 45.0),
                           "note": "ideal values use Js = 16.1 kG; RPI's E sits ~2 % above (kappa); for magnetically "
                                   "independent grains isotropic <cos>=0.5 caps Br near Js/2, and 45 MGOe needs Br >= "
                                   "13.4 kG, i.e. alignment within these models (exchange-enhanced isotropic remanence "
                                   "above Js/2 is not modelled)"}
        res["scenarios"] = _scenarios(r)
        res["feedstock"] = self._feedstock_info(r)
        res["flag_rules"] = ("prior_dominated: posterior-to-prior predictive spread ratio (IQR^2 on the transformed "
                             "scale) > 0.5, or the output's module is out of its data hull; mechanism_dominated: > 50 % "
                             "of the predictive variance lies between mechanism-conditional means")
        res["objective"] = OBJECTIVE
        if rho_observed is not None:
            res["rho_observed_note"] = ("magnetics conditioned on rho_observed as a MEASURED density (fixed 0.005 relative "
                                        "spread); for a body not yet made pass rho_scenario={'mean', 'sd'} instead")
        res["recording_note"] = ("record a forecast with its quantiles, not mean +/- sd: the magnetic predictives are "
                                 "skewed mixtures; mu_ln / sd_ln summarise them only where lognormal_adequate is true; "
                                 "forecast_record() returns a ready-to-store record")
        if include_draws:
            res["draws"] = {k: np.asarray(v).tolist() for k, v in sims.items() if not str(k).startswith("_")}
        return self._stamp(res)

    # ---------------------------------------------------------------- C7 joint target on both magnetic paths
    JOINT_ASSUMPTIONS = {
        "c5": {"Hci": "C5 mechanism average (H0-H5, load-timing explanations) fitted on historical A, C, D, E and E's "
                      "anneals; isotropic fits; smooth ceiling 37.5 kOe",
               "Br": "Js rho <cos theta> (1 + kappa) R(Hci) x E's fitted anneal factor x (body's anneal density gain "
                     "/ E's gain)",
               "squareness": "logit fitted on isotropic E curves (and E's anneals); does not rise with alignment",
               "BHmax": "parametric second-quadrant curve x fitted shape factor, capped at Br^2/4",
               "alignment": "scales Br only", "temperature": "declared scenario (Kuz'min Js(T), linear HA(T), Neff from "
                                                             "the body's grain)"},
        "link": {"Hci": "alpha mu0H_N - Neff Js, alpha anchored on historical E (7.5 kOe at E's grain and placement), "
                        "coverage from the Nd-rich placement latents, boundary-film magnetization J_GB in {0, 0.6, 1.2} T "
                        "with its Phase 2a weights, C5 chemistry/porosity offset, declared along-axis penalty for "
                        "alignment; body scatter borrowed from C5 (sig_spec, sig_piece)",
                 "Br": "Js rho <cos theta> (1 - f_Nd) (1 + kappa) R(Hci) at the stage's own density (C7), no fitted "
                       "anneal factor",
                 "squareness": "Hk/Hci from an isotropic range (0.22-0.31) to an aligned one (0.80-0.95) by <cos theta>",
                 "BHmax": "same parametric curve, capped at Br^2/4", "alignment": "Br, Hci penalty and squareness",
                 "temperature": "the same declared scenario as C5, applied to the body's link Hci and Br"}}

    def joint_target(self, route, spec=None, n_bodies=3, n_preparations=1, n_worlds=1500, bodies_per_world=12,
                     seed=0, rho_observed=None, rho_scenario=None, density_stage="as_sintered"):
        """Probability that the bodies of a route meet a joint spec (default the objective: Hci >= 20 kOe and (BH)max >=
        45 MGOe), on both magnetic paths from one set of draws. A world theta is one joint draw of everything no body
        varies: the SPS density draw, the preparation effect and degassing state of each preparation, the coercivity
        mechanism and load-timing explanation, H0's direction, the C6 microstructure-layer draw, the anneal draw, the
        coercivity link's own epistemic draws (intrinsic constants, E anchor, coverage, anneal re-wetting, alignment
        penalty, squareness) with one boundary-film magnetization J_GB, and the temperature scenario. Bodies vary by
        density scatter, specimen and piece effects and the Br, squareness and curve-shape scatter (C5's fitted terms;
        the link borrows them). p(theta) is a body's pass probability; P(all n bodies of one preparation) = E[p^n],
        estimated without bias from bodies_per_world bodies; P(every body of n_preparations preparations) uses a fresh
        preparation effect per preparation. The product P(one body)^n that treats bodies as independent is reported
        only to show the dependence. spec keys as p_meets_spec (T_operating_C applies the declared temperature map to
        both paths). The route's anneal key sets the stage."""
        spec = dict(OBJECTIVE_SPEC if spec is None else _validate_spec(dict(spec)))
        if not (self._has_c6() and self.link is not None):
            raise ValueError("joint_target needs the C6 microstructure layer and the coercivity link")
        for nm, v, lo in (("n_bodies", n_bodies, 1), ("n_preparations", n_preparations, 1), ("n_worlds", n_worlds, 100),
                          ("bodies_per_world", bodies_per_world, 2)):
            if not (isinstance(v, int) and v >= lo):
                raise ValueError(f"{nm} must be an integer >= {lo}")
        if n_bodies > bodies_per_world:
            raise ValueError("bodies_per_world must be at least n_bodies")
        route = dict(route)
        if "T_operating_C" in spec:
            route["T_measure_C"] = float(spec["T_operating_C"])
        r = self._route(route)
        resolution = r.pop("_resolution", None)
        r["_degas_expected"] = self._expected_state(r)
        A = self._arrays(r)
        W, M, K = n_worlds, n_preparations, bodies_per_world
        N = W * M * K
        rng = np.random.default_rng([int(seed), 89])
        world = np.repeat(np.arange(W), M * K)            # world of each body
        prep = np.repeat(np.arange(W * M), K)              # (world, preparation) block of each body
        start = np.arange(W) * M * K                       # first body of each world
        dlab_w, didx_w = self._assign_bma("density", W, rng)
        mlab_w, midx_w = self._assign_bma("magnetic", W, rng)
        dlab, didx, mlab, midx = dlab_w[world], didx_w[world], mlab_w[world], midx_w[world]
        ind = {k: self._assign_indicator(k, mlab_w, midx_w, rng)[world] for k in self._indicators()}
        zprep = rng.standard_normal(W * M)[prep][:, None]
        if rho_observed is not None or rho_scenario is not None:
            if rho_observed is not None and rho_scenario is not None:
                raise ValueError("pass either rho_observed or rho_scenario, not both")
            if rho_observed is not None:
                v = float(rho_observed)
                mu, sd = (v / RHO_TD if v > 1.5 else v), 0.005
            else:
                mu, sd = self._rho_scenario(rho_scenario)
            rho_mean_w = np.full(W, mu)
            rho = np.clip(mu + sd * rng.standard_normal(N), 0.3, 1.02)
        else:
            mu, sig = self._density_mean(A, r, dlab, didx, zprep, "post")
            rho_mean_w = mu[start]
            rho = np.clip(mu + sig * rng.standard_t(5.0, N), 0.3, 1.02)
        degas_p, eps_p = self._degas_state(r, rng.random(W * M), np.random.default_rng([int(seed), 89, 31]))
        degas_ind = degas_p[prep]
        eps = None if eps_p is None else eps_p[prep]
        z_hull = rng.standard_normal(W)[world]
        sig_spec = np.array([self.post[m]["sig_spec"][i] for m, i in zip(mlab, midx)])
        u = sig_spec * rng.standard_normal(N)
        lat = self._magnetic_latent(A, r, mlab, midx, rho, degas_ind, z_hull, u, "post", eps=eps, indicators=ind,
                                    sem=True)
        c7 = None
        if self._has_c7():
            amech_w, ap_w = self._c7_draws(W, rng)
            c7 = self._c7_state(r, rho, lat, mlab, midx, "post", rng, density_stage,
                                draws=(amech_w[world], {k: v[world] for k, v in ap_w.items()}))
        if c7 is not None:   # C7.1: H4's grain term follows the hold, as in predict()
            lat["lh"] = self._c7_apply_lh_shift(lat["lh"], c7["h4_shift"])
        stage = {"none": "as_sintered", "900": "after_900", "900+520": "after_900_520"}[r["anneal"]]
        rho_stage = np.asarray(c7["stage_rho"][stage], float) if c7 is not None else rho
        T_C = float(r["T_measure_C"])
        s_w, cHA_w = rng.uniform(0.5, 0.6, W)[world], rng.uniform(0.0026, 0.0032, W)[world]

        def at_temperature(hci, br, lat_):
            if abs(T_C - 20.0) < 1e-9:
                return hci, br
            raw, rJ = self._temperature_map(lat_, hci, T_C, s_w, cHA_w)
            hci_T = np.maximum(raw, 0.0)
            R = lambda h: 1.0 / (1.0 + (lat_["HR"] / np.maximum(h, 1e-3)) ** lat_["kR"])
            return hci_T, br * rJ * R(hci_T) / R(hci)

        # ---- C5 path (the predict() measurement model, world-level shared terms as above)
        z_piece, z_br, z_sq = (rng.standard_normal(N) for _ in range(3))
        hci5 = np.exp(lat["lh"] + lat["sig_piece"] * z_piece)
        br5 = np.exp(lat["lbr"] + lat["sig_Br"] * z_br)
        if c7 is not None and c7["br_factor"] is not None:
            br5 = br5 * c7["br_factor"]
        sq5 = 1.0 / (1.0 + np.exp(-(lat["sq"] + lat["sig_sq"] * z_sq)))
        hci5, br5 = at_temperature(hci5, br5, lat)
        bh5 = self._bh(br5, hci5, sq5, rng)
        # ---- coercivity link: one evaluation per world (its epistemic draws and J_GB shared by the world's bodies)
        ms = self.ms
        P6 = self._c6_params(mlab_w, midx_w, "post")
        lat_w = {k: np.asarray(v)[start] for k, v in lat.items()}
        ox = self._c6_oxide_factor(r, lat_w, mlab_w, midx_w, "post")
        a3s = ms.a3_latent(P6, lat_w["W_bse"], stage, ox)
        a3a = ms.a3_latent(P6, lat_w["W_bse"], "as_sintered", ox)
        if c7 is not None and "grain_stage" in c7:
            g_w = np.asarray(c7["grain_stage"][stage], float)[start]
        else:
            g_w = lat_w["g_as_sintered"] if stage == "as_sintered" else lat_w["g_after_900_520"]
        gain_w = (np.asarray(c7["stage_rho"][stage], float)[start] / np.asarray(c7["stage_rho"]["as_sintered"],
                                                                                 float)[start]
                  if c7 is not None else np.ones(W))
        rho_link_w = np.clip(rho_mean_w * gain_w, 0.3, 1.02)
        flat = {"grain_median_um": g_w, "grain_sigma_ln": P6["c6_sig"], "density_rel": rho_link_w,
                "nd_rich_area_fraction": a3s["nd_rich_area_fraction"], "pocket_share": a3s["pocket_share"],
                "faint_line_coverage": a3s["faint_line_coverage"], "pocket_share_as_sintered": a3a["pocket_share"],
                "faint_line_coverage_as_sintered": a3a["faint_line_coverage"],
                "coverage_sem_lower": np.full(W, np.nan), "stage": np.full(W, stage),
                "alignment_cos_theta": np.full(W, float(r[ALIGN_KEY])),
                "ln_hci_chem_offset": lat_w["ln_chem_offset"]}
        core = getattr(self.link, "_link_flat", self.link.link)
        lo = core(flat, np.random.default_rng([int(seed), 89, 67]))
        levels = list(self.link.JGB_LEVELS)
        jw = np.array([self.link.JGB_POSTERIOR[J] for J in levels], float)
        jpick_w = rng.choice(len(levels), size=W, p=jw / jw.sum())
        stack = lambda key: np.stack([np.asarray(lo[key]["by_jgb"][J], float) for J in levels])
        hci_w = stack("hci_kOe")[jpick_w, np.arange(W)]
        br_w = stack("br_kGs")[jpick_w, np.arange(W)]
        hk_w = np.asarray(lo["hk_over_hci"]["by_jgb"][levels[0]], float)
        hciL = hci_w[world] * np.exp(u + lat["sig_piece"] * z_piece)
        brL = br_w[world] * (rho_stage / rho_link_w[world]) * np.exp(lat["sig_Br"] * z_br)
        hciL, brL = at_temperature(hciL, brL, lat)
        bhL = np.minimum(self.link._bhmax(brL, np.maximum(hciL, 1e-3), hk_w[world]), brL ** 2 / 4.0)
        r.pop("_degas_expected", None)

        keys = (("Hci_kOe_min", "hci"), ("BHmax_MGOe_min", "bh"), ("Br_kGs_min", "br"), ("density_rel_min", "rho"))

        def summarize(vals, extra):
            ok = np.ones(N, bool)
            marg = {}
            for key, v in keys:
                if key in spec:
                    c = np.asarray(vals[v]) >= float(spec[key])
                    marg[key] = float(c.mean())
                    ok &= c
            s_blk = ok.reshape(W * M, K).sum(1).astype(float)          # passing bodies per (world, preparation)
            ff = np.ones(W * M)
            for j in range(n_bodies):                                   # unbiased estimate of p^n per block
                ff *= np.maximum(s_blk - j, 0.0) / (K - j)
            p_all = ff.reshape(W, M).prod(1)
            p_w = ok.reshape(W, M * K).mean(1)
            q = np.quantile(p_w, [0.05, 0.25, 0.5, 0.75, 0.95])
            p1 = float(ok.mean())
            out = {"p_one_body": p1, "p_one_body_se": float(np.std(p_w, ddof=1) / math.sqrt(W)),
                   f"p_all_{n_bodies}_bodies_x_{n_preparations}_preparation": float(p_all.mean()),
                   "p_all_se": float(np.std(p_all, ddof=1) / math.sqrt(W)),
                   "independent_bodies_product_not_valid": p1 ** (n_bodies * n_preparations),
                   "p_world_quantiles": dict(zip(("q05", "q25", "median", "q75", "q95"), map(float, q))),
                   "share_of_worlds_with_p_above_0.5": float(np.mean(p_w > 0.5)),
                   "marginals": marg,
                   "by_magnetic_mechanism": {m: float(ok[mlab == m].mean()) for m in np.unique(mlab)},
                   "no_body_meets_spec": bool(p1 == 0.0),
                   "p_one_body_upper_95_if_none": (3.0 / W) if p1 == 0.0 else None,
                   "medians": {k: float(np.median(vals[v])) for k, v in (("Hci_kOe", "hci"), ("Br_kGs", "br"),
                                                                          ("BHmax_MGOe", "bh"))}}
            out.update(extra(ok))
            return out

        c5 = summarize({"hci": hci5, "br": br5, "bh": bh5, "rho": rho_stage}, lambda ok: {})
        lk = summarize({"hci": hciL, "br": brL, "bh": bhL, "rho": rho_stage}, lambda ok: {
            "by_jgb_T": {f"{levels[j]:g}": float(ok[jpick_w[world] == j].mean()) for j in range(len(levels))
                         if np.any(jpick_w == j)},
            "link_flags_share": {k: float(np.mean(v)) for k, v in lo["flags"].items()}})
        need = math.sqrt(4.0 * float(spec["BHmax_MGOe_min"])) if "BHmax_MGOe_min" in spec else None
        cos = float(r[ALIGN_KEY])
        structural = {"Br_needed_kGs": need,
                      "Br_bound_kGs_Js_rho_cos": 16.1 * float(np.median(rho_stage)) * cos * 1.02,
                      "share_Br_at_least_needed": None if need is None else {
                          "c5": float(np.mean(br5 >= need)), "link": float(np.mean(brL >= need))},
                      "note": ("(BH)max <= Br^2/4 (CGS: kG, MGOe) on any second-quadrant curve. In both paths an "
                               "isotropic body (<cos theta> = 0.5) of magnetically independent grains has Br near Js rho "
                               "/ 2 ~ 8 kG, so within these models (BH)max <= ~16 MGOe whatever its coercivity and the "
                               "45 MGOe leg needs alignment. This is a model bound, not a universal one: exchange-coupled "
                               "isotropic nanocrystalline and nanocomposite magnets reach remanence above Js/2, which "
                               "neither path represents (and which is not a demonstrated route to the joint target)"),
                      "assumptions": "Stoner-Wohlfarth remanence of non-interacting randomly oriented grains; "
                                     "Js = 16.1 kG at room temperature; no exchange enhancement"}
        res = {"route": r, "route_resolution": resolution, "spec": spec, "stage": stage,
               "n_bodies": n_bodies, "n_preparations": n_preparations, "n_worlds": W, "bodies_per_world": K,
               "paths": {"c5": c5, "coercivity_link": lk}, "assumptions": self.JOINT_ASSUMPTIONS,
               "structural": structural, "out_of_hull": self.hull(r, rho=float(np.median(rho))),
               "scenarios": _scenarios(r), "feedstock": self._feedstock_info(r),
               "dependence": ("bodies of one preparation share theta (parameters, preparation effect, degassing state, "
                              "J_GB, link and anneal draws, temperature scenario); P(all) = E[p(theta)^n]. Paths share "
                              "the density, grain, Nd-rich placement, chemistry and C5 scatter draws, so their "
                              "agreement is not independent evidence"),
               "status": ("conditional model probabilities: no current-panel magnetic measurement exists; aligned, "
                          "hot-temperature and fine-feedstock conditions are declared extrapolations; a path's zero is "
                          "a property of that model at this route, not a physical impossibility unless the structural "
                          "bound decides it"),
               "c7": self._c7_block(r, c7, seed) if c7 is not None else None}
        return self._stamp(res)

    def p_meets_spec(self, route, spec, n_draws=6000, seed=0, rho_observed=None, rho_scenario=None,
                     density_stage="as_sintered"):
        """Posterior probability that one body made by `route` meets `spec`, scored on *measured* values (the same
        measurement scatter as the RPI curves). spec keys (all optional): Hci_kOe_min, BHmax_MGOe_min, Br_kGs_min,
        density_rel_min, T_operating_C (evaluates Hci/Br/BHmax at that temperature as a declared scenario).
        C7: this is the C5 magnetic path for one body; joint_target() reports both magnetic paths together, with the
        dependence between the bodies of one preparation."""
        _validate_spec(spec)
        route = dict(route)
        if "T_operating_C" in spec:
            route["T_measure_C"] = float(spec["T_operating_C"])
        r, sims, dlab, mlab, hull, ind = self._simulate(route, n_draws, seed, "post", rho_observed, rho_scenario,
                                                        density_stage=density_stage)
        resolution = r.pop("_resolution", None)
        keys = (("Hci_kOe_min", "Hci_kOe"), ("BHmax_MGOe_min", "BHmax_MGOe"), ("Br_kGs_min", "Br_kGs"),
                ("density_rel_min", "density_rel"))
        ok = np.ones(n_draws, bool)
        checks = {}
        for key, out in keys:
            if key in spec:
                c = np.asarray(sims[out]) >= float(spec[key])
                checks[key] = float(c.mean())
                ok &= c
        p = float(ok.mean())
        by_m = {m: float(ok[mlab == m].mean()) for m in np.unique(mlab)}
        _, prs, _, _, _, _ = self._simulate(route, max(2000, n_draws // 2), seed + 1, "prior", rho_observed,
                                            rho_scenario, density_stage=density_stage)
        okp = np.ones(len(prs["Hci_kOe"]), bool)
        for key, out in keys:
            if key in spec:
                okp &= np.asarray(prs[out]) >= float(spec[key])
        mag_spec = any(k in spec for k in ("Hci_kOe_min", "BHmax_MGOe_min", "Br_kGs_min"))
        spread = max(by_m.values()) - min(by_m.values()) if by_m else 0.0
        res = {"route": r, "spec": spec, "p_meets_spec": p,
               "monte_carlo_se": math.sqrt(max(p * (1 - p), 1e-12) / n_draws), "marginals": checks,
               "by_magnetic_mechanism": by_m, "mechanism_weights": self.w["magnetic"],
               "by_load_timing": ({L: float(ok[ind["load_timing"] == L].mean()) for L in LT_LEVELS
                                   if (ind["load_timing"] == L).any()} if "load_timing" in ind else None),
               "load_timing_weights": self.w.get("load_timing"),
               "prior_predictive_p": float(okp.mean()),
               "prior_dominated": bool((mag_spec and hull["magnetic"]["out_of_hull"])
                                       or (hull["density"]["out_of_hull"] and "density_rel_min" in spec)
                                       or (abs(p - okp.mean()) < 0.02 and 0.0 < p < 1.0) or spread > 0.3),
               "no_draw_meets_spec": bool(p == 0.0 and float(okp.mean()) == 0.0),
               "p_upper_95": (3.0 / n_draws) if p == 0.0 else None,
               "out_of_hull": hull, "scenario_not_calibrated": not _scenarios(r)["calibrated"],
               "feedstock": self._feedstock_info(r),
               "scenarios": _scenarios(r),
               "scored_on": "measured values (piece, Br, squareness and curve-shape scatter included)",
               "route_resolution": resolution, "magnetic_path": "C5 (one body); see joint_target() for both paths"}
        return self._stamp(res)

    def coarse_number_fraction(self, route, coarse_threshold_nm=None, stage="as_sintered", n_draws=4000, seed=0,
                               rho_observed=None, rho_scenario=None):
        """C6: number fraction of grains larger than coarse_threshold_nm (nm; required) at `stage` (as_sintered,
        after_900, after_900_520) under the SEM-protocol grain distribution, as predict() computes it."""
        if not self._has_c6():
            raise ValueError("this package carries no C6 microstructure layer")
        thr = self._check_threshold(coarse_threshold_nm)
        stage = self.ms.stage6(stage)
        res = self.predict(route, n_draws=n_draws, seed=seed, rho_observed=rho_observed, rho_scenario=rho_scenario,
                           coarse_threshold_nm=thr)
        g = res["outputs"]["grain_sem_dist"]
        out = {"route": res["route"], "stage": stage, "coarse_threshold_nm": thr,
               "coarse_number_fraction": g[stage]["coarse_fraction"], "status": g["status"],
               "status_note": g["status_note"], "measurement_model": g["measurement_model"]}
        return self._stamp(out)

    def sps_log(self, route, n_draws=4000, seed=0, rho_observed=None, rho_scenario=None, n_traces=1, dt_s=1.0,
                band_dt_s=10.0, charge_mass_g=None):
        """C6 (A4): a descriptive SPS display trace for the requested schedule: temperature on the route's ramp and
        dwell (Newtonian cooling at the route's initial cooling rate after power-off), relative density from a
        pre-load ramp densification (onset 505-540 C) plus the load step (at power-off for load on cooling; at the
        requested time, declared, otherwise), and the ram displacement derived from the geometry (10 mm die, a
        declared charge, tooling expansion and compliance), quantized like RPI's logs (1 s, 0.1 C, 0.01 mm). The end
        state of draw i is exactly the C5 density draw i of predict(route, n_draws, seed) with the same arguments.
        Returns n_traces traces at dt_s and q05/q50/q95 bands over all draws at band_dt_s. Descriptive only: the
        densification kinetics are not identified."""
        if not self._has_c6():
            raise ValueError("this package carries no C6 microstructure layer")
        if not (isinstance(n_traces, int) and 0 <= n_traces <= min(int(n_draws), 20)):
            raise ValueError("n_traces must be an integer in [0, min(n_draws, 20)]")
        for nm, v in (("dt_s", dt_s), ("band_dt_s", band_dt_s)):
            if not (_is_number(v) and 1.0 <= float(v) <= 60.0):
                raise ValueError(f"{nm} must be a number of seconds in [1, 60]")
        ms = self.ms
        mass = ms.SPS_CHARGE_MASS_G if charge_mass_g is None else float(charge_mass_g)
        if not 0.2 <= mass <= 3.0:
            raise ValueError("charge_mass_g must lie in [0.2, 3.0] g (RPI's #1-#5: 0.77-0.92 g in a 10 mm die)")
        # the density draws come first in _simulate, so they equal predict()'s for the same arguments
        r, sims, dlab, mlab, hull, ind = self._simulate(route, n_draws, seed, "post", rho_observed, rho_scenario)
        r.pop("_resolution", None)
        rho = np.asarray(sims.get("density_rel_as_sintered", sims["density_rel"]), float)  # the SPS trace ends as sintered
        prm = ms.sps_draw(len(rho), np.random.default_rng([int(seed), 61, 4]))
        band = ms.sps_trace(r, rho, prm, dt_s=float(band_dt_s), mass_g=mass, quantize=False)
        q = (0.05, 0.5, 0.95)
        bands = {"t_s": band["t_s"].tolist(), "T_C": band["T_C"].tolist(),
                 **{f"{ch}_{k}": np.quantile(band[ch], lv, axis=0).tolist()
                    for ch in ("rel_density", "displacement_mm") for k, lv in zip(("q05", "q50", "q95"), q)}}
        traces = []
        if n_traces:
            sub = {k: v[:n_traces] for k, v in prm.items()}
            tr = ms.sps_trace(r, rho[:n_traces], sub, dt_s=float(dt_s), mass_g=mass, quantize=True)
            for i in range(n_traces):
                ev = {k: (float(v[i]) if np.ndim(v) else float(v)) for k, v in tr["events"].items()}
                traces.append({"draw": i, "t_s": tr["t_s"].tolist(), "T_C": tr["T_C"].tolist(),
                               "displacement_mm": tr["displacement_mm"][i].tolist(),
                               "rel_density": tr["rel_density"][i].tolist(), "end_rel_density": float(rho[i]),
                               "events": ev})
        ev = band["events"]
        summ = lambda x: dict(median=float(np.median(x)), q05=float(np.quantile(x, 0.05)), q95=float(np.quantile(x, 0.95)))
        declared = r["pressure_mode"] != "cooling" or abs(float(r["load_ramp_min"]) - 0.3) > 1e-9
        res = {"route": r, "status": ms.SPS_STATUS if r["pressure_mode"] == "cooling"
               else ms.SPS_STATUS + "; " + ms.SPS_STATUS_DECLARED_TIMING,
               "declared_scenario": bool(declared),
               "status_note": ("relative density is derived from the geometry (RPI's logs carry time, displacement and "
                               "temperature only; no pressure, force or current channel); the end state is the C5 "
                               "density draw; the shape (onset, pre-load share, step share and duration) follows "
                               "declared ranges from RPI's logs #1, #2, #4, #5 (#3's temperature duplicates #2's); no "
                               "log exists above 800 C, so the shape at 850 C is an extrapolation"
                               + ("; a load ramp other than 0.3 min scales the step duration (declared)"
                                  if abs(float(r["load_ramp_min"]) - 0.3) > 1e-9 else "")),
               "end_density_rel": summ(rho), "step_dD_mm": summ(ev["step_dD_mm"]),
               "step_duration_s": summ(ev["step_duration_s"]), "pre_load_points_TD": summ(ev["pre_load_points_TD"]),
               "step_share": summ(ev["step_share"]), "t_power_off_s": ev["t_power_off_s"],
               "band": bands, "traces": traces, "n_draws": int(n_draws), "seed": int(seed),
               "quantization": dict(ms.SPS_QUANTIZATION), "geometry": {"die_diameter_mm": ms.SPS_DIE_DIAMETER_MM,
                                                                      "charge_mass_g": mass,
                                                                      "theoretical_density_g_cm3": ms.SPS_RHO_TD},
               "channels": {"t_s": "s", "T_C": "deg C (sensor reading)", "displacement_mm": "ram displacement, mm "
                            "(increases with shrinkage)", "rel_density": "fraction of 7.62 g/cm3 (derived)"}}
        return self._stamp(res)

    def forecast_record(self, route, body_id=None, rho_observed=None, n_draws=8000, seed=0,
                        outputs=("density_rel", "Hci_kOe", "Br_kGs", "BHmax_MGOe", "Hk_over_Hci"), rho_scenario=None,
                        coarse_threshold_nm=None, density_stage="as_sintered"):
        """A schema-ready committed-forecast record for one body (deterministic for a given seed, so it can be stored
        and reproduced). body_id: a neutral caller-supplied identifier. Contains, per output, the quantile set and
        log-normal parameters with the flags; the resolved route and scenario keys; the conditioning; and the tool
        and artifact hashes. The scoring harness reads this record directly. C6: rec["c6"] carries the microstructure
        outputs in the same per-output schema (coarse number fractions only with coarse_threshold_nm)."""
        res = self.predict(route, n_draws=n_draws, seed=seed, rho_observed=rho_observed, rho_scenario=rho_scenario,
                           coarse_threshold_nm=coarse_threshold_nm, density_stage=density_stage)
        rec_out = {}
        for k in outputs:
            o = res["outputs"][k]
            rec_out[k] = {"quantiles": o["quantiles"], "mu_ln": o["mu_ln"], "sd_ln": o["sd_ln"],
                          "lognormal_adequate": o["lognormal_adequate"],
                          "lognormal_max_log_error": o["lognormal_max_log_error"], "median": o["median"],
                          "frac_nonpositive": o["frac_nonpositive"], "prior_dominated": o["prior_dominated"],
                          "mechanism_dominated": o["mechanism_dominated"],
                          "posterior_to_prior_spread_ratio": o["posterior_to_prior_spread_ratio"],
                          "conditioned_on_observed_density": o["conditioned_on_observed_density"],
                          "status": o["status"]}
        rec = {"schema": "ndfeb-forecast-record-v1", "body_id": body_id, "route": res["route"],
                "status": res["feedstock"]["status"], "feedstock": res["feedstock"],
                "scenarios": res["scenarios"], "conditioning": {"rho_observed": rho_observed,
                                                                "rho_scenario": rho_scenario},
                "outputs": rec_out, "out_of_hull": res["out_of_hull"], "mechanism_weights": res["mechanism_weights"],
                "n_draws": int(n_draws), "seed": int(seed), "quantile_levels": list(QUANTILE_LEVELS),
                "units": {"density_rel": "fraction of 7.62 g/cm3", "Hci_kOe": "kOe", "Br_kGs": "kG",
                          "BHmax_MGOe": "MGOe", "Hk_over_Hci": "ratio"},
                "scoring": "score from the quantiles (interpolated predictive on the log scale for positive quantities); "
                           "mu_ln / sd_ln only where lognormal_adequate; 50/90 % interval coverage",
                "tool_version": TOOL_VERSION, "evidence_class": EVIDENCE_CLASS, "artifact_sha256": self.hashes,
                "artifact_verified": self.verified, "conditioned_on": self.conditioning,
                "route_resolution": res.get("route_resolution"), "density_stage": density_stage,
                "c7": {k: res["c7"].get(k) for k in ("route_stage", "anneal_hold_h", "anneal_mechanism_weights",
                                                     "stages") if k in res["c7"]}}
        if "gb_phase_bse" in res["outputs"]:  # C5: the SEM/BSE outputs (continuous ones in the per-output schema)
            o = res["outputs"]
            sem = {}
            for key, src, st in (("grain_size_sem_um_as_sintered", o["grain_size_sem_um"]["as_sintered"],
                                  o["grain_size_sem_um"]["status"]),
                                 ("grain_size_sem_um_after_900_520", o["grain_size_sem_um"]["after_900_520"],
                                  o["grain_size_sem_um"]["status"]),
                                 ("anneal_grain_change_um", o["anneal_grain_change"]["delta_um"],
                                  o["anneal_grain_change"]["status"]),
                                 ("anneal_grain_change_ratio", o["anneal_grain_change"]["ratio"],
                                  o["anneal_grain_change"]["status"])):
                sem[key] = {"quantiles": src["quantiles"], "mu_ln": src["mu_ln"], "sd_ln": src["sd_ln"],
                            "lognormal_adequate": src["lognormal_adequate"],
                            "lognormal_max_log_error": src["lognormal_max_log_error"], "median": src["median"],
                            "frac_nonpositive": src["frac_nonpositive"], "prior_dominated": src["prior_dominated"],
                            "mechanism_dominated": src["mechanism_dominated"],
                            "posterior_to_prior_spread_ratio": src["posterior_to_prior_spread_ratio"], "status": st}
            for s in SEM_STAGES:
                g = o["gb_phase_bse"][s]
                sem["gb_phase_bse_" + s] = {"p_spread": g["p_spread"], "p_pooled": g["p_pooled"], "stage": s,
                                            "status": g["status"]}
            rec["sem_bse"] = {"outputs": sem,
                              "status_notes": {**{k: o[k]["status_note"] for k in SEM_OUTPUTS},
                                               "gb_phase_bse_after_900_520": o["gb_phase_bse"]["after_900_520"]["status_note"]},
                              "units": {"grain_size_sem_um_as_sintered": "um", "grain_size_sem_um_after_900_520": "um",
                                        "anneal_grain_change_um": "um", "anneal_grain_change_ratio": "ratio",
                                        "gb_phase_bse_as_sintered": "probability of a 'spread' reading (else 'pooled')",
                                        "gb_phase_bse_after_900_520": "probability of a 'spread' reading (else 'pooled')"},
                              "scoring": "continuous outputs from the quantiles as above; gb_phase_bse_* as binary "
                                         "forecasts (p_spread)"}
        if "fracture_intercept_um" in res["outputs"]:  # C6: the microstructure outputs in the per-output schema
            rec["c6"] = self._c6_record(res, coarse_threshold_nm)
        return rec

    @staticmethod
    def _rec_entry(src, status):
        if src is None:
            return {"value": None, "status": status}
        return {"quantiles": src["quantiles"], "mu_ln": src["mu_ln"], "sd_ln": src["sd_ln"],
                "lognormal_adequate": src["lognormal_adequate"], "lognormal_max_log_error": src["lognormal_max_log_error"],
                "median": src["median"], "frac_nonpositive": src["frac_nonpositive"],
                "prior_dominated": src["prior_dominated"], "mechanism_dominated": src["mechanism_dominated"],
                "posterior_to_prior_spread_ratio": src["posterior_to_prior_spread_ratio"], "status": status}

    def _c6_record(self, res, coarse_threshold_nm):
        o = res["outputs"]
        out, units = {}, {}
        for s in C6_STAGES:
            fi, gd = o["fracture_intercept_um"], o["grain_sem_dist"]
            out["fracture_intercept_um_" + s] = self._rec_entry(fi[s], fi["status"])
            out["grain_median_um_" + s] = self._rec_entry(gd[s]["median_um"], gd["status"])
            out["grain_d90_um_" + s] = self._rec_entry(gd[s]["d90_um"], gd["status"])
            units.update({"fracture_intercept_um_" + s: "um", "grain_median_um_" + s: "um", "grain_d90_um_" + s: "um"})
            if coarse_threshold_nm is not None:
                out["coarse_number_fraction_" + s] = self._rec_entry(gd[s]["coarse_fraction"], gd["status"])
                units["coarse_number_fraction_" + s] = f"number fraction above {coarse_threshold_nm:g} nm"
            for k in self.ms.A3_KEYS:
                out[f"{k}_{s}"] = self._rec_entry(o[k][s], o[k]["status"])
                units[f"{k}_{s}"] = "um" if k == "pocket_ecd_um" else "fraction"
        p = o["porosity_area_fraction_sem"]
        out["porosity_area_fraction_sem"] = self._rec_entry(p["value"], p["status"])
        units["porosity_area_fraction_sem"] = "area fraction"
        return {"outputs": out, "units": units, "coarse_threshold_nm": coarse_threshold_nm,
                "status_notes": {k: o[k]["status_note"] for k in ("grain_sem_dist", "fracture_intercept_um",
                                                                  "porosity_area_fraction_sem") + tuple(self.ms.A3_KEYS)},
                "p_planar": p["p_planar"], "alignment": res["c6"]["alignment"],
                "coercivity_link_status": o["coercivity_link"].get("status"),
                "scoring": "continuous outputs from the quantiles; value None (status not_planar) is a no call"}

    def _latents(self, meas, target, dlab, didx, mlab, midx, rng, indicators=None):
        """Per-route latent means and noise scales for a set of joint draws (aligned arrays of length n). Shared across
        routes within a draw: new-preparation effects, the degas state of each preparation, H0's discrepancy direction."""
        n = len(mlab)
        routes = [mm["route"] for mm in meas] + [target]
        rho_obs = [mm.get("rho_observed") for mm in meas] + [None]
        prep_z, degas, eps, prep_dc, prep_fd = {}, {}, {}, {}, {}
        z_hull = rng.standard_normal(n)
        rows = []
        extra = None  # child generator, created only when a new v3.1 option needs randomness (keeps v3 equivalence)
        for j, route in enumerate(routes):
            r = self._route(route)
            r["_degas_expected"] = self._expected_state(r)
            A = self._arrays(r)
            if r["prep"] in prep_dc and prep_dc[r["prep"]] != json.dumps(r["degas_completeness"], sort_keys=True):
                raise ValueError(f"routes sharing preparation {r['prep']!r} must share degas_completeness")
            prep_dc[r["prep"]] = json.dumps(r["degas_completeness"], sort_keys=True)
            _check_prep_feedstock(prep_fd, r)
            if r["prep"] not in prep_z:
                prep_z[r["prep"]] = rng.standard_normal((n, 1))
                u = rng.random(n)
                if not self._completeness_is_default(r) and extra is None:
                    extra = rng.spawn(1)[0]
                degas[r["prep"]], eps[r["prep"]] = self._degas_state(r, u, extra)
            mu, sig = self._density_mean(A, r, dlab, didx, prep_z[r["prep"]])
            ro = rho_obs[j]
            rho = np.full(n, (ro / RHO_TD if ro > 1.5 else ro)) if ro else np.clip(mu, 0.3, 1.0)
            lat = self._magnetic_latent(A, r, mlab, midx, rho, degas[r["prep"]], z_hull, eps=eps[r["prep"]],
                                        indicators=indicators)
            row = dict(mu_rho=mu, sig_rho=sig, rho_known=ro is not None, **lat)
            if j < len(meas) and "Hci_ratio_T" in meas[j].get("quantities", []):
                if extra is None:
                    extra = rng.spawn(1)[0]
                row["lratioT"], row["lratioT_g"] = self._ln_hci_ratio_T(lat, float(meas[j].get("T2_C", 120.0)), extra)
            rows.append(row)
        return rows

    def _ln_hci_ratio_T(self, lat, T2_C, rng):
        """Declared measurement hook (uncalibrated): ln[Hci(T2) / Hci(20 C)] at the route-level latent under the tool's
        temperature scenario (the same map as predict), with per-draw Kuz'min s and HA(T) slope; floored at ln(0.01)
        (a loop whose coercivity collapses is field-limited, not exact). Returns the ratio and its sensitivity
        g = d ln(ratio) / d ln(Hci): the specimen effect does not cancel, because alpha follows the body's own Hci, so
        the eig noise adds (g * sig_spec)^2 to 2 * sig_piece^2."""
        n = len(lat["lh"])
        s = rng.uniform(0.5, 0.6, n)
        cHA = rng.uniform(0.0026, 0.0032, n)

        def ln_ratio(lh):
            hci = np.exp(lh)
            raw, _ = self._temperature_map(lat, hci, T2_C, s, cHA)
            return np.log(np.maximum(raw, 0.01 * hci) / hci)
        r0 = ln_ratio(lat["lh"])
        g = (ln_ratio(lat["lh"] + 0.01) - r0) / 0.01
        return r0, g

    # ---------------------------------------------------------------- experiment design
    def _eig_v31(self, candidates, n_outer=1500, n_inner_per_mechanism=6000, seed=0, target=None):
        """The v3.1-r3 estimator, unchanged (eig(..., method="nmc-v3.1")). Rank candidate measurements by expected
        information gain (nats).
        candidates: list of {"name": str, "measurements": [ {"route": {...}, "rho_observed": optional,
                    "quantities": ["Hci","Br","sq","density","grain","H_ppm","Hci_ratio_T"], "replicates": 1,
                    "T2_C": 120 (only for Hci_ratio_T)} ]}
        H_ppm and Hci_ratio_T use declared, uncalibrated measurement models (H_MEASUREMENT_LOG_SD; the temperature
        scenario), so their EIG is conditional on those assumptions.
        Mechanism EIG is estimated two ways from the same draws: nested Monte Carlo with the outer draw excluded
        (consistent; biased low when the inner set is small) and with the outer draw included in its own stratum
        (biased high for informative designs). The reported value is the midpoint of that bracket; `*_converged` is
        False when the bracket is wider than max(0.05 nats, 2 SE): then increase n_inner_per_mechanism. The
        all-parameter EIG is a prior-contrastive-style estimate (outer draw included; cf. Foster et al. 2019). The
        density-mechanism EIG counts the density measurements only (0 when no density is measured; a lower bound when
        magnetics are measured on a body of unknown density, which also carries density information). The midpoint
        uses max(low, 0), since a negative lower estimate is small-sample bias. p_identify is bracketed the same way;
        expected posterior weights come from the outer-draw-included estimator. Also: the
        expected posterior mechanism weights, the probability that the MAP mechanism is the generating one, and the
        expected change of the predictive sd of ln Hci at `target` (a route; default the first measurement's)."""
        out = []
        for ci, cand in enumerate(candidates):
            out.append(self._eig_one_v31(cand, n_outer, n_inner_per_mechanism,
                                         np.random.default_rng(seed + 7919 * (ci + 1)), target))
        for d in out:
            d["ln_Hci_sd_reduction_target"] = d["ln_Hci_sd_target_now"] - d["ln_Hci_sd_target_expected_after"]
        orderings = {key: [d["name"] for d in sorted(out, key=lambda d: -d[key])]
                     for key in ("eig_magnetic_mechanism_nats", "eig_density_mechanism_nats", "eig_all_parameters_nats",
                                 "ln_Hci_sd_reduction_target")}
        out.sort(key=lambda d: (-d["eig_magnetic_mechanism_nats"], -d["eig_all_parameters_nats"]))
        for i, d in enumerate(out):
            d["rank"] = i + 1
        return self._stamp({"ranking": out, "ranking_rule": "primary: EIG on the magnetic mechanism indicator "
                            "(bracket midpoint); tie-break: EIG on all parameters; see orderings for every metric",
                            "orderings": orderings,
                            "all_converged": bool(all(d["eig_magnetic_mechanism_converged"]
                                                      and d["eig_density_mechanism_converged"] for d in out)),
                            "method": "nested Monte Carlo, stratified inner sets per magnetic mechanism; mechanism "
                                      "EIG bracketed by excluding / including the outer draw in its stratum; Gaussian "
                                      "measurement models in log space (Student-t(5) for density); per-body specimen "
                                      "effects integrated; H0 discrepancy direction, new-preparation effects and "
                                      "degas state shared across routes within a draw"})

    def _eig_one_v31(self, cand, n_outer, n_per, rng, target, chunk=6000):
        meas = cand["measurements"]
        tgt = target or meas[0]["route"]
        dl_o, di_o = self._assign_bma("density", n_outer, rng)
        ml_o, mi_o = self._assign_bma("magnetic", n_outer, rng)
        ml_i, mi_i = self._assign_strata("magnetic", n_per, rng)
        n_inner = len(ml_i)
        dl_i, di_i = self._assign_bma("density", n_inner, rng)
        dlab, didx = np.concatenate([dl_o, dl_i]), np.concatenate([di_o, di_i])
        mlab, midx = np.concatenate([ml_o, ml_i]), np.concatenate([mi_o, mi_i])
        ind = {k: self._assign_indicator(k, mlab, midx, rng) for k in self._indicators()}
        rows = self._latents(meas, tgt, dlab, didx, mlab, midx, rng, indicators=ind)
        o = slice(0, n_outer)
        H_SD = H_MEASUREMENT_LOG_SD  # declared: log-sd of an H-content measurement relative to the desorption index
        # simulated outcomes, one per outer draw; per-term inner means and scales
        terms, own, own_d = [], np.zeros(n_outer), np.zeros(n_outer)
        for j, mm in enumerate(meas):
            R = rows[j]
            q = set(mm.get("quantities", ["Hci", "Br", "sq"]))
            reps = int(mm.get("replicates", 1))
            if "density" in q and not R["rho_known"]:
                y = R["mu_rho"][o] + R["sig_rho"][o] * rng.standard_t(5.0, n_outer)
                terms.append((_t_logpdf, y, R["mu_rho"][n_outer:], R["sig_rho"][n_outer:], True))
                own += _t_logpdf(y, R["mu_rho"][o], R["sig_rho"][o])
                own_d += _t_logpdf(y, R["mu_rho"][o], R["sig_rho"][o])
            spec = []
            if "Hci" in q:  # the body's specimen effect integrated: ln Hci ~ N(lh, sig_spec^2 + sig_piece^2/reps)
                spec.append(("lh", np.sqrt(R["sig_spec"] ** 2 + R["sig_piece"] ** 2 / reps)))
            if "Br" in q:
                spec.append(("lbr", R["sig_Br"] / math.sqrt(reps)))
            if "sq" in q:
                spec.append(("sq", R["sig_sq"] / math.sqrt(reps)))
            if "grain" in q:
                spec.append(("lgrain", np.full(len(R["lh"]), 0.15)))
            if "H_ppm" in q:
                spec.append(("lh_res", np.full(len(R["lh"]), H_SD)))
            if "Hci_ratio_T" in q:  # same body at two temperatures; delta-method specimen term (alpha follows the body)
                spec.append(("lratioT", np.sqrt(2.0 * R["sig_piece"] ** 2 / reps + (R["lratioT_g"] * R["sig_spec"]) ** 2)))
            ys = {}
            for key, sd in spec:
                if key == "lratioT" and "lh" in ys:
                    # Hci and Hci_ratio_T on the same body share its specimen effect u and the 20 C piece noise e20:
                    # ln Hci = lh + u + e20, ln ratio = r0 + g u + (eT - e20), so cov = g sig_spec^2 - sig_piece^2/reps.
                    # Modelled jointly as ln ratio | ln Hci (a conditional normal), not as an independent term.
                    v1 = R["sig_spec"] ** 2 + R["sig_piece"] ** 2 / reps
                    cov = R["lratioT_g"] * R["sig_spec"] ** 2 - R["sig_piece"] ** 2 / reps
                    cslope = cov / v1
                    sdc = np.sqrt(np.maximum(sd ** 2 - cov ** 2 / v1, 1e-12))
                    y1 = ys["lh"]
                    y = R[key][o] + cslope[o] * (y1 - R["lh"][o]) + sdc[o] * rng.standard_normal(n_outer)
                    terms.append((_n_logpdf, y, R[key][n_outer:], sdc[n_outer:], False,
                                  (y1, R["lh"][n_outer:], cslope[n_outer:])))
                    own += _n_logpdf(y, R[key][o] + cslope[o] * (y1 - R["lh"][o]), sdc[o])
                    continue
                y = R[key][o] + sd[o] * rng.standard_normal(n_outer)
                ys[key] = y
                terms.append((_n_logpdf, y, R[key][n_outer:], sd[n_outer:], False))
                own += _n_logpdf(y, R[key][o], sd[o])
        res = {"name": cand.get("name", ""), "measurements": meas, "cost": cand.get("cost"),
               "n_outer": n_outer, "n_inner_per_magnetic_mechanism": n_per}
        kinds = {}
        for kind, lab_o, lab_i in (("magnetic", ml_o, ml_i), ("density", dl_o, dl_i)):
            mechs = [m for m in self._mechs(kind) if self.w[kind][m] > 0]
            w = np.array([self.w[kind][m] for m in mechs])
            kinds[kind] = (mechs, w / w.sum(), lab_o, lab_i, np.full((n_outer, len(mechs)), -np.inf))
        # target-route moments (importance weights over the inner set, re-weighted to the model average)
        T = rows[-1]
        wbase = np.array([self.w["magnetic"][m] / (ml_i == m).sum() for m in ml_i])
        tgt_inner = T["lh"][n_outer:]
        M = np.full(n_outer, -np.inf)
        S0, S1, S2, Sq = (np.zeros(n_outer) for _ in range(4))
        for c0 in range(0, n_inner, chunk):
            c1 = min(n_inner, c0 + chunk)
            ll = np.zeros((n_outer, c1 - c0))
            ll_d = np.zeros((n_outer, c1 - c0))
            for tm in terms:
                f, y, mu, sd, is_density = tm[:5]
                if len(tm) > 5:  # conditional term: mean shifts with the paired observation y1 (same body)
                    y1, mu1, cs = tm[5]
                    mu_eff = mu[None, c0:c1] + cs[None, c0:c1] * (y1[:, None] - mu1[None, c0:c1])
                    term = f(y[:, None], mu_eff, sd[None, c0:c1])
                else:
                    term = f(y[:, None], mu[None, c0:c1], sd[None, c0:c1])
                ll += term
                if is_density:
                    ll_d += term
            for kind, (mechs, w, lab_o, lab_i, acc) in kinds.items():
                lab = lab_i[c0:c1]
                src = ll if kind == "magnetic" else ll_d
                for k, m in enumerate(mechs):
                    cols = lab == m
                    if cols.any():
                        acc[:, k] = np.logaddexp(acc[:, k], _lse(src[:, cols], 1))
            lw = ll + np.log(wbase[c0:c1])[None, :]
            cmax = np.max(lw, 1)
            newM = np.maximum(M, cmax)
            scale = np.exp(np.where(np.isfinite(M), M - newM, -np.inf))
            ew = np.exp(lw - newM[:, None])
            tt = tgt_inner[c0:c1]
            S0 = S0 * scale + ew.sum(1)
            S1 = S1 * scale + ew @ tt
            S2 = S2 * scale + ew @ (tt ** 2)
            Sq = Sq * scale ** 2 + (ew ** 2).sum(1)
            M = newM
        lpy_mag = None
        has_density = any(term[4] for term in terms)
        for kind, (mechs, w, lab_o, lab_i, acc) in kinds.items():
            idx = np.array([mechs.index(l) for l in lab_o])
            own_k = own if kind == "magnetic" else own_d
            est = {}
            for include_own in (False, True):
                logp = np.empty((n_outer, len(mechs)))
                for k, m in enumerate(mechs):
                    n_m = int((lab_i == m).sum())
                    mine = lab_o == m
                    if include_own:
                        logp[:, k] = (np.where(mine, np.logaddexp(acc[:, k], own_k), acc[:, k])
                                      - np.log(np.maximum(n_m + mine, 1)))
                    else:
                        logp[:, k] = acc[:, k] - math.log(max(n_m, 1))
                lpy = _lse(logp + np.log(w)[None, :], 1)
                contrib = logp[np.arange(n_outer), idx] - lpy
                ok = np.isfinite(contrib)
                n_ok = max(int(ok.sum()), 2)
                post = np.exp(logp + np.log(w)[None, :] - lpy[:, None])
                pid = float(np.mean((np.argmax(post, 1) == idx)[ok])) if ok.any() else float("nan")
                est[include_own] = (float(contrib[ok].mean()) if ok.any() else float("nan"),
                                    float(contrib[ok].std(ddof=1) / math.sqrt(n_ok)) if ok.sum() > 1 else float("nan"),
                                    post, lpy, pid, int((~ok).sum()))
            lo, hi = est[False][0], est[True][0]
            se = max(est[False][1], est[True][1])
            pid_lo, pid_hi = est[False][4], est[True][4]
            if kind == "density" and not has_density:  # nothing measured that the density mechanism predicts
                lo = hi = se = 0.0
            # the true EIG is >= 0; a negative lower estimate is small-sample bias, so the midpoint uses max(lo, 0)
            res[f"eig_{kind}_mechanism_nats"] = 0.5 * (max(lo, 0.0) + hi)
            res[f"eig_{kind}_mechanism_bracket"] = [lo, hi]
            res[f"eig_{kind}_mechanism_se"] = se
            res[f"eig_{kind}_mechanism_converged"] = bool(abs(hi - lo) <= max(0.05, 2 * se))
            res[f"n_nonfinite_outer_{kind}"] = max(est[False][5], est[True][5])
            post = est[True][2]
            res[f"expected_posterior_{kind}_weights"] = {m: float(post[:, k].mean()) for k, m in enumerate(mechs)}
            res[f"p_identify_{kind}_mechanism_map"] = 0.5 * (pid_lo + pid_hi)
            res[f"p_identify_{kind}_mechanism_bracket"] = [pid_lo, pid_hi]
            if kind == "magnetic":
                lpy_mag = est[True][3]
        contrib = own - lpy_mag
        res["eig_all_parameters_nats"] = float(contrib.mean())
        res["eig_all_parameters_se"] = float(contrib.std(ddof=1) / math.sqrt(n_outer))
        w0 = wbase / wbase.sum()
        now_var = float(w0 @ tgt_inner ** 2 - (w0 @ tgt_inner) ** 2)
        spec_var = float(w0 @ T["sig_spec"][n_outer:] ** 2)
        m1, m2 = S1 / S0, S2 / S0
        res["target_route"] = tgt
        res["ln_Hci_sd_target_now"] = math.sqrt(now_var + spec_var)
        res["ln_Hci_sd_target_expected_after"] = float(np.mean(np.sqrt(np.maximum(m2 - m1 ** 2, 0) + spec_var)))
        res["ess_inner_median"] = float(np.median(S0 ** 2 / Sq))
        return res

    # ---------------------------------------------------------------- experiment design (v4 estimator)
    def eig(self, candidates, n_outer=1500, n_inner_per_mechanism=None, seed=0, target=None, goal=None,
            method="v4"):
        """Rank candidate measurements by expected information gain (nats); see the eig section of the README.
        candidates: list of {"name": str, "measurements": [ {"route": {...}, "rho_observed": optional,
                    "quantities": ["Hci","Br","sq","density","grain","H_ppm","Hci_ratio_T"], "replicates": 1,
                    "T2_C": 120 (only for Hci_ratio_T)} ]}; each measurement is one body.
        "grain" is the model grain with a field log-sd of 0.15 and no SEM method factor c_m, so it overstates the
        information in an SEM grain report (C5; predict's grain_size_sem_um carries c_m).
        For every per-draw discrete indicator the model carries (self.w: magnetic and density mechanisms, and any
        other, e.g. load timing) the EIG is bracketed by two estimates from the same simulated experiments:
          lower: a Barber-Agakov bound, H(prior weights) + E[log q(true label | y)], where q pools the exact nested
                 posterior over an inner set of stored draws that shares no stored draw with the simulated experiment
                 (weight rising with its effective sample size) with a classifier trained on simulated experiments
                 from the other half of the stored draws (cross-fitted), mixed with the prior at EIG_EPS. It is a
                 lower bound on the EIG in expectation whatever the quality of q;
          upper: the nested estimate with the generating draw included in its own stratum (as in v3.1, biased high
                 when the stored draws cannot resolve the design).
        The value (`*_nats`) is max(lower, 0): the design gains at least this much information about the indicator.
        `*_lower_bound_stable` (= `*_converged`): the two halves (independent simulated experiments, each scored by a
        classifier trained on the other half's stored draws) agree within 2 combined SE. `*_bracket_closed`: upper -
        lower <= max(0.05, 2 SE), so the value is the EIG itself (single-body designs); for multi-body designs the
        upper end stays loose because the stored posterior holds about 2,000 draws per mechanism.
        goal: optional {"route": {...}, "spec": {...p_meets_spec keys...}, "rho_observed": optional} adds the
        information about reaching the spec at that route (see `_eig_goal`).
        method="nmc-v3.1" runs the v3.1-r3 estimator unchanged (no goal, magnetic and density only)."""
        if method == "nmc-v3.1":
            if goal is not None:
                raise ValueError("goal needs method='v4'")
            extra = self._eig_indicator_names()[1]
            if extra:
                raise ValueError(f"method='nmc-v3.1' predates per-draw indicators beyond the mechanisms ({extra}); "
                                 "use method='v4'")
            return self._eig_v31(candidates, n_outer, n_inner_per_mechanism or 6000, seed, target)
        n_inner_per_mechanism = n_inner_per_mechanism or 4000
        if method != "v4":
            raise ValueError("method must be 'v4' (default) or 'nmc-v3.1' (the v3.1-r3 estimator, for comparison)")
        if not candidates:
            raise ValueError("eig needs at least one candidate")
        goal = self._eig_goal_parse(goal) if goal is not None else None
        out = [self._eig_one(cand, n_outer, n_inner_per_mechanism, np.random.default_rng(seed + 7919 * (ci + 1)),
                             target, goal) for ci, cand in enumerate(candidates)]
        names = list(out[0]["eig_indicators"])
        keys = [self._eig_key(n) + "_nats" for n in names] + ["eig_all_parameters_nats", "ln_Hci_sd_reduction_target"]
        if goal is not None:
            keys += ["eig_goal_event_nats", "goal_p_success_sd_reduction"]
        orderings = {k: [d["name"] for d in sorted(out, key=lambda d, k=k: -d[k])] for k in keys}
        out.sort(key=lambda d: (-d["eig_magnetic_mechanism_nats"], -d["eig_all_parameters_nats"]))
        for i, d in enumerate(out):
            d["rank"] = i + 1
        return self._stamp({
            "ranking": out, "orderings": orderings, "indicators": names,
            "ranking_rule": "primary: EIG on the magnetic mechanism indicator; tie-break: EIG on all parameters; see "
                            "orderings for every metric",
            "all_converged": bool(all(d["eig_indicators"][n]["lower_bound_stable"] for d in out for n in names)),
            "converged_rule": "value = the nested estimate H(prior) - E[H(posterior)], clipped into [lower bound, "
                              "upper], when the bracket is closed, "
                              "else max(lower bound, 0); lower_bound_stable (= converged, and all_converged over "
                              "every indicator and candidate): the two independent halves agree within 2 combined SE; "
                              "bracket_closed: upper - lower <= max(0.05 nats, 2 SE), i.e. the value is the EIG itself; "
                              "rank differences smaller than about 3 lower-bound SE are not resolved",
            "method": EIG_METHOD_V4 + ": simulated experiments from the stored posterior draws (two halves by stored-"
                      "draw parity); exact likelihoods of each simulated experiment under an inner set that excludes "
                      "its own stored draws (Gaussian measurement models in log space, Student-t(5) for density, "
                      "per-body specimen effects integrated; new-preparation effects, degas state, H0 discrepancy "
                      "direction and conditional indicators shared across routes within a draw); per indicator a "
                      "Barber-Agakov lower bound with the pooled posterior (exact nested posterior where its effective "
                      "sample size allows, cross-fitted classifier elsewhere) and the outer-draw-included nested "
                      "estimate as the upper end"})

    @staticmethod
    def _eig_key(name):
        return f"eig_{name}_mechanism" if name in ("magnetic", "density") else f"eig_{name}"

    def _eig_indicator_names(self):
        """(base, conditional) indicators. Base: every positive-weight label has stored parameter arrays (magnetic,
        density); eig assigns them. Conditional: labels drawn per magnetic draw by self._assign_indicator."""
        base, cond = [], []
        for name, wd in self.w.items():
            labs = [l for l, v in wd.items() if v > 0]
            (base if labs and all(l in self.post for l in labs) else cond).append(name)
        if "magnetic" not in base or "density" not in base:
            raise ValueError("eig needs the magnetic and density indicators with stored draws")
        base.sort(key=lambda n: (n != "magnetic", n != "density", n))
        return base, cond

    def _eig_labels(self, name):
        return [l for l, v in self.w[name].items() if v > 0]

    def _eig_draws(self, n, rng, balanced):
        """n joint draws. Base indicators: labels in equal shares (balanced, the inner set) or in proportion to the
        weights (the simulated experiments), each carrying the log importance weight log(w / share); stored indices by
        systematic resampling on the per-draw weights, restricted to one parity ("fold") per joint draw so the two
        halves share no stored draw. Conditional indicators: self._assign_indicator(name, mlab, midx, rng)."""
        base, cond = self._eig_indicator_names()
        fold = rng.permutation(np.arange(n) % 2)
        D = {"n": n, "fold": fold, "lab": {}, "idx": {}, "logw_own": {}, "logw": np.zeros(n)}
        for name in base:
            labs = self._eig_labels(name)
            w = np.array([self.w[name][l] for l in labs], float)
            w = w / w.sum()
            cnt = _alloc(w, n, balanced)
            code = np.repeat(np.arange(len(labs)), cnt)[rng.permutation(n)]
            lw = np.log(w)[code] - np.log(np.maximum(cnt, 1) / n)[code]
            idx = np.zeros(n, int)
            for k, l in enumerate(labs):
                K = len(next(iter(self.post[l].values())))
                for f in (0, 1):
                    pos = np.where((code == k) & (fold == f))[0]
                    if not len(pos):
                        continue
                    # C5: folds by the parity of each draw's first stored copy, so copies of one draw (a resampled
                    # posterior) stay in one half; without copies this is exactly np.arange(f, K, 2)
                    cand = np.where(self._first_index(l) % 2 == f)[0] if K > 1 else np.arange(K)
                    p = np.asarray(self.dw[l], float)[cand]
                    p = p if p.sum() > 0 else np.ones(len(cand))
                    idx[pos] = cand[_systematic(p, len(pos), rng)][rng.permutation(len(pos))]
            D["lab"][name], D["idx"][name], D["logw_own"][name] = np.array(labs)[code], idx, lw
            D["logw"] = D["logw"] + lw
        for name in cond:
            D["lab"][name] = np.asarray(self._assign_indicator(name, D["lab"]["magnetic"], D["idx"]["magnetic"], rng)
                                        ).astype(str)
            D["logw_own"][name] = np.zeros(n)
        return D

    @staticmethod
    def _eig_cat(A, B):
        cat = lambda a, b: np.concatenate([a, b])
        return {"n": A["n"] + B["n"], "fold": cat(A["fold"], B["fold"]), "logw": cat(A["logw"], B["logw"]),
                "lab": {k: cat(A["lab"][k], B["lab"][k]) for k in A["lab"]},
                "idx": {k: cat(A["idx"][k], B["idx"][k]) for k in A["idx"]},
                "logw_own": {k: cat(A["logw_own"][k], B["logw_own"][k]) for k in A["logw_own"]}}

    def _eig_measurements(self, cand):
        meas = cand.get("measurements") if isinstance(cand, dict) else None
        if not meas:
            raise ValueError("each candidate needs a non-empty 'measurements' list")
        out = []
        for mm in meas:
            q = list(mm.get("quantities", ["Hci", "Br", "sq"]))
            bad = [x for x in q if x not in EIG_QUANTITIES]
            if bad or not q:
                raise ValueError(f"unknown or empty quantities {bad}; allowed: {list(EIG_QUANTITIES)}")
            reps = mm.get("replicates", 1)
            if not (_is_number(reps) and int(reps) == reps and reps >= 1):
                raise ValueError("replicates must be a positive integer")
            r = self._route(mm["route"])
            if r["anneal"] != "none" and self._has_c7():   # C7.1: eig's latents carry no anneal layer
                staged = sorted(set(q) & {"density", "Br", "grain"})
                if staged:
                    raise ValueError(f"eig cannot score {staged} at an annealed stage: its latents carry no anneal "
                                     "densification, stage density or hold-scaled grain (C7.1). Use anneal_eig() for "
                                     "density readings after an anneal, joint_target() for magnetic outcomes")
                if r.get(ANNEAL_HOLD_KEY) != ANNEAL_HOLD_DEFAULT:
                    raise ValueError("eig scores annealed-stage Hci/Hk only at the 2 h hold of E's fitted program "
                                     "(C7.1: H4's hold-scaled grain is not in eig's latents)")
            if abs(float(r["T_measure_C"]) - 20.0) > 1e-9 and set(q) & {"Hci", "Br", "sq", "Hci_ratio_T"}:
                raise ValueError("eig models Hci, Br and Hk/Hci measured at 20 C: a route with T_measure_C != 20 is "
                                 "not supported for these quantities; measure the hot loop with Hci_ratio_T and T2_C")
            out.append(dict(mm, quantities=q, replicates=int(reps)))
        return out

    def _eig_latents(self, meas, extra_routes, D, rng):
        """Latent rows for the draws in D, one row per distinct (route, rho_observed); shared within a draw across
        routes: new-preparation effects, the degas state of each preparation, H0's discrepancy direction and the
        conditional indicators. Hci_ratio_T hooks are per measurement (as in v3.1)."""
        n = D["n"]
        mlab, midx = D["lab"]["magnetic"], D["idx"]["magnetic"]
        dlab, didx = D["lab"]["density"], D["idx"]["density"]
        cond = self._eig_indicator_names()[1]
        kw = {"indicators": {k: D["lab"][k] for k in cond}} if cond else {}
        z_hull = rng.standard_normal(n)
        prep_z, degas, eps, prep_dc, prep_fd = {}, {}, {}, {}, {}
        extra = None
        rows, rkey, mrow, xrow, ratio, rres = [], {}, [], [], {}, []
        items = [(mm["route"], mm.get("rho_observed")) for mm in meas] + [(rt, None) for rt in extra_routes]
        for j, (route, ro) in enumerate(items):
            r = self._route(route)
            key = json.dumps([r, ro], sort_keys=True, default=str)
            if key not in rkey:
                dc = json.dumps(r["degas_completeness"], sort_keys=True, default=str)
                if prep_dc.setdefault(r["prep"], dc) != dc:
                    raise ValueError(f"routes sharing preparation {r['prep']!r} must share degas_completeness")
                _check_prep_feedstock(prep_fd, r)
                r["_degas_expected"] = self._expected_state(r)
                A = self._arrays(r)
                if r["prep"] not in prep_z:
                    prep_z[r["prep"]] = rng.standard_normal((n, 1))
                    u = rng.random(n)
                    if not self._completeness_is_default(r) and extra is None:
                        extra = rng.spawn(1)[0]
                    degas[r["prep"]], eps[r["prep"]] = self._degas_state(r, u, extra)
                mu, sig = self._density_mean(A, r, dlab, didx, prep_z[r["prep"]])
                rho = np.full(n, (ro / RHO_TD if ro > 1.5 else ro)) if ro else np.clip(mu, 0.3, 1.0)
                lat = self._magnetic_latent(A, r, mlab, midx, rho, degas[r["prep"]], z_hull, eps=eps[r["prep"]], **kw)
                rkey[key] = len(rows)
                rows.append(dict(mu_rho=mu, sig_rho=sig, rho_known=ro is not None, **lat))
                rres.append(r)
            (mrow if j < len(meas) else xrow).append(rkey[key])
        for j, mm in enumerate(meas):
            if "Hci_ratio_T" in mm["quantities"]:
                if extra is None:
                    extra = rng.spawn(1)[0]
                ratio[j] = self._ln_hci_ratio_T(rows[mrow[j]], float(mm.get("T2_C", 120.0)), extra)
        return {"rows": rows, "routes": rres, "mrow": mrow, "xrow": xrow, "ratio": ratio, "kw": kw,
                "nuis": {"z_hull": z_hull, "prep_z": prep_z, "degas": degas, "eps": eps}}

    @staticmethod
    def _eig_obs(meas, L):
        """Observation models, one entry per measured quantity per body (arrays over all draws):
        kind "g": Gaussian in log space (mean mu, sd), optionally conditional on the same body's Hci (pair, cs);
        kind "t": Student-t(5) density. col groups observations of one latent for the summaries."""
        obs = []
        for j, mm in enumerate(meas):
            rk = L["mrow"][j]
            R = L["rows"][rk]
            q, reps = set(mm["quantities"]), mm["replicates"]
            if "density" in q and not R["rho_known"]:
                obs.append(dict(kind="t", mu=R["mu_rho"], sd=R["sig_rho"], pair=-1, cs=None, col=(rk, "density")))
            hci = -1
            for qn, key, sd in (("Hci", "lh", lambda: np.sqrt(R["sig_spec"] ** 2 + R["sig_piece"] ** 2 / reps)),
                                ("Br", "lbr", lambda: R["sig_Br"] / math.sqrt(reps)),
                                ("sq", "sq", lambda: R["sig_sq"] / math.sqrt(reps)),
                                ("grain", "lgrain", lambda: np.full(len(R["lh"]), 0.15)),
                                ("H_ppm", "lh_res", lambda: np.full(len(R["lh"]), H_MEASUREMENT_LOG_SD))):
                if qn in q:
                    obs.append(dict(kind="g", mu=R[key], sd=sd(), pair=-1, cs=None, col=(rk, key)))
                    if qn == "Hci":
                        hci = len(obs) - 1
            if "Hci_ratio_T" in q:  # same body at two temperatures (v3.1-r3 joint model with its Hci)
                r0, g = L["ratio"][j]
                sd = np.sqrt(2.0 * R["sig_piece"] ** 2 / reps + (g * R["sig_spec"]) ** 2)
                if hci >= 0:
                    v1 = R["sig_spec"] ** 2 + R["sig_piece"] ** 2 / reps
                    cov = g * R["sig_spec"] ** 2 - R["sig_piece"] ** 2 / reps
                    obs.append(dict(kind="g", mu=r0, sd=np.sqrt(np.maximum(sd ** 2 - cov ** 2 / v1, 1e-12)),
                                    pair=hci, cs=cov / v1, col=("ratio", j)))
                else:
                    obs.append(dict(kind="g", mu=r0, sd=sd, pair=-1, cs=None, col=("ratio", j)))
        return obs

    @staticmethod
    def _eig_simulate(obs, sel, rng):
        n = sel.stop - sel.start
        Y = np.empty((n, len(obs)))
        for t, ob in enumerate(obs):
            mu, sd = ob["mu"][sel], ob["sd"][sel]
            if ob["kind"] == "t":
                Y[:, t] = mu + sd * rng.standard_t(5.0, n)
            elif ob["pair"] >= 0:
                p = ob["pair"]
                Y[:, t] = mu + ob["cs"][sel] * (Y[:, p] - obs[p]["mu"][sel]) + sd * rng.standard_normal(n)
            else:
                Y[:, t] = mu + sd * rng.standard_normal(n)
        return Y

    @staticmethod
    def _eig_loglik_own(obs, sel, Y):
        """Log-likelihood of each simulated outcome under its own draw (the rows of Y align with sel)."""
        ll = np.zeros(sel.stop - sel.start)
        for t, ob in enumerate(obs):
            mu, sd = ob["mu"][sel], ob["sd"][sel]
            if ob["kind"] == "t":
                ll += _t_logpdf(Y[:, t], mu, sd)
            else:
                m = mu + (ob["cs"][sel] * (Y[:, ob["pair"]] - obs[ob["pair"]]["mu"][sel]) if ob["pair"] >= 0 else 0.0)
                ll += _n_logpdf(Y[:, t], m, sd)
        return ll

    @staticmethod
    def _eig_summaries(obs, Y):
        """Classifier inputs: per latent, the mean of its body observations; where >= 2 bodies, the log sd."""
        cols = {}
        for t, ob in enumerate(obs):
            cols.setdefault(ob["col"], []).append(t)
        M = [Y[:, ts].mean(1) for ts in cols.values()]
        V = [np.log(Y[:, ts].std(1, ddof=1) + 1e-3) for ts in cols.values() if len(ts) >= 2]
        return np.stack(M + V, 1), len(M)

    @staticmethod
    def _eig_gauss_terms(obs, sp, center):
        """Gaussian log densities of outcomes under the draws in sp as feat(Y) @ C + c0, where feat(Y) =
        [Y - c, (Y - c)^2, cross products of conditional pairs] over the Gaussian observations (one matrix product)."""
        gi = [t for t, ob in enumerate(obs) if ob["kind"] == "g"]
        pos = {t: k for k, t in enumerate(gi)}
        pairs = [t for t in gi if obs[t]["pair"] >= 0]
        G, n = len(gi), sp.stop - sp.start
        C, c0 = np.zeros((2 * G + len(pairs), n)), np.zeros(n)
        for k, t in enumerate(gi):
            ob = obs[t]
            w = 1.0 / ob["sd"][sp] ** 2
            kap = ob["mu"][sp] - center[t]
            if ob["pair"] >= 0:
                p, cs = ob["pair"], ob["cs"][sp]
                kap = kap - cs * (obs[p]["mu"][sp] - center[p])
                C[G + pos[p]] += -0.5 * cs * cs * w
                C[2 * G + pairs.index(t)] += cs * w
                C[pos[p]] += -cs * kap * w
            C[k] += kap * w
            C[G + k] += -0.5 * w
            c0 += -0.5 * kap * kap * w - np.log(ob["sd"][sp]) - 0.5 * math.log(2 * math.pi)

        def feat(Y):
            Yc = Y[:, gi] - center[gi]
            cr = [((Y[:, t] - center[t]) * (Y[:, obs[t]["pair"]] - center[obs[t]["pair"]]))[:, None] for t in pairs]
            return np.concatenate([Yc, Yc ** 2] + cr, 1)
        return feat, C, c0

    def _eig_nmc(self, obs, YO, sp, keysO, keysP, logwP, H, X, chunk_o=400, chunk_p=4000):
        """Exact likelihood of every simulated outcome (rows of YO) under every inner draw (sp), leaving out inner
        draws that share a stored draw with the outcome's own draw (keys). Returns log sums of w f per label column of
        H, of (w f)^2 per column, of w f over all inner draws, and posterior means of the columns of X."""
        no, npool = YO.shape[0], sp.stop - sp.start
        center = YO.mean(0)
        feat, C, c0 = self._eig_gauss_terms(obs, sp, center)
        F = feat(YO)
        ti = [t for t, ob in enumerate(obs) if ob["kind"] == "t"]
        tmu = [obs[t]["mu"][sp] for t in ti]
        tinv = [1.0 / obs[t]["sd"][sp] for t in ti]
        tc = math.lgamma(3.0) - math.lgamma(2.5) - 0.5 * math.log(5.0 * math.pi)
        c0 = c0 + sum(tc - np.log(obs[t]["sd"][sp]) for t in ti)
        L, nx = H.shape[1], X.shape[1]
        out = {"logS": np.empty((no, L)), "logS2": np.empty((no, L)), "logSall": np.empty(no),
               "mean_x": np.empty((no, nx))}
        # the left-out entries (copies of the outcome's own stored draws) are kept apart: adding them back gives the
        # v3.1-style estimate in which the generating stored draw is part of the inner set ("stored-draw" estimate)
        mo, mp, mv = [], [], []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for o0 in range(0, no, chunk_o):
                os_ = slice(o0, min(no, o0 + chunk_o))
                m = os_.stop - os_.start
                M = np.full(m, -np.inf)
                S, S2, SX, Sall = np.zeros((m, L)), np.zeros((m, L)), np.zeros((m, nx)), np.zeros(m)
                for p0 in range(0, npool, chunk_p):
                    ps = slice(p0, min(npool, p0 + chunk_p))
                    ll = F[os_] @ C[:, ps] + c0[ps]
                    for k, t in enumerate(ti):
                        z = (YO[os_, t][:, None] - tmu[k][ps][None]) * tinv[k][ps][None]
                        z *= z
                        z *= 0.2
                        ll -= 3.0 * np.log1p(z)
                    mask = np.zeros(ll.shape, bool)
                    for kO, kP in zip(keysO, keysP):
                        mask |= kO[os_][:, None] == kP[ps][None, :]
                    r_, c_ = np.nonzero(mask)
                    if len(r_):
                        mo.append(r_ + o0)
                        mp.append(c_ + p0)
                        mv.append(ll[r_, c_] + logwP[ps][c_])
                    ll[mask] = -np.inf
                    ll += logwP[ps][None]
                    newM = np.maximum(M, ll.max(1))
                    shift = np.where(np.isfinite(newM), newM, 0.0)
                    sc = np.where(np.isfinite(M), np.exp(M - shift), 0.0)
                    E = np.exp(ll - shift[:, None])
                    S = S * sc[:, None] + E @ H[ps]
                    S2 = S2 * (sc * sc)[:, None] + (E * E) @ H[ps]
                    SX = SX * sc[:, None] + E @ X[ps]
                    Sall = Sall * sc + E.sum(1)
                    M = newM
                out["logS"][os_] = np.log(S) + M[:, None]
                out["logS2"][os_] = np.log(S2) + 2 * M[:, None]
                out["logSall"][os_] = np.log(Sall) + M
                out["mean_x"][os_] = SX / Sall[:, None]
        logSm, logSm_all = np.full((no, L), -np.inf), np.full(no, -np.inf)
        if mo:
            mo, mp, mv = np.concatenate(mo), np.concatenate(mp), np.concatenate(mv)
            np.logaddexp.at(logSm_all, mo, mv)
            Hb = H[mp] > 0
            for j in range(L):
                sel = Hb[:, j]
                np.logaddexp.at(logSm[:, j], mo[sel], mv[sel])
        out["logS_copies"], out["logSall_copies"] = logSm, logSm_all
        return out

    def _eig_structural_zero(self, name, meas, extra_routes, rng_seed):
        """True when no measured quantity depends on the indicator: latents recomputed for 64 draws with only this
        indicator's labels (and, for a base indicator, its stored draw) changed are identical."""
        D = self._eig_draws(64, np.random.default_rng(rng_seed), balanced=True)
        labs = self._eig_labels(name)
        if len(labs) < 2:
            return True
        D2 = {**D, "lab": dict(D["lab"]), "idx": dict(D["idx"])}
        new = np.array([labs[(labs.index(l) + 1) % len(labs)] for l in D["lab"][name]])
        D2["lab"][name] = new
        if name in D["idx"]:
            r = np.random.default_rng(rng_seed + 1)
            D2["idx"][name] = np.array([r.integers(len(next(iter(self.post[l].values())))) for l in new])
        o1 = self._eig_obs(meas, self._eig_latents(meas, extra_routes, D, np.random.default_rng(rng_seed + 2)))
        o2 = self._eig_obs(meas, self._eig_latents(meas, extra_routes, D2, np.random.default_rng(rng_seed + 2)))
        return all(np.array_equal(a["mu"], b["mu"]) and np.array_equal(a["sd"], b["sd"]) for a, b in zip(o1, o2))

    def _eig_one(self, cand, n_outer, n_per, rng, target, goal=None):
        meas = self._eig_measurements(cand)
        tgt = target or meas[0]["route"]
        base, cond = self._eig_indicator_names()
        names = base + cond
        n_mag = len(self._eig_labels("magnetic"))
        N, n_outer = max(4, int(n_per)) * n_mag, max(40, int(n_outer))
        extra_routes = [tgt] + ([goal["route"]] if goal is not None else [])
        zero = {nm: self._eig_structural_zero(nm, meas, extra_routes, int(rng.integers(2 ** 31))) for nm in names}
        DP, DO = self._eig_draws(N, rng, balanced=True), self._eig_draws(n_outer, rng, balanced=False)
        D = self._eig_cat(DP, DO)
        L = self._eig_latents(meas, extra_routes, D, rng)
        obs = self._eig_obs(meas, L)
        sp, so = slice(0, N), slice(N, N + n_outer)
        YO, YP = self._eig_simulate(obs, so, rng), self._eig_simulate(obs, sp, rng)
        own = self._eig_loglik_own(obs, so, YO)
        # label columns of every indicator, leave-one-out keys of every stored draw, posterior-mean columns
        cols, H = {}, []
        for nm in names:
            labs = self._eig_labels(nm)
            start = sum(h.shape[1] for h in H)
            H.append((D["lab"][nm][sp][:, None] == np.array(labs)[None, :]).astype(float))
            cols[nm] = list(range(start, start + len(labs)))
        H = np.concatenate(H, 1)
        keys = []
        for nm in base:
            labs = self._eig_labels(nm)
            code = np.array([labs.index(l) for l in D["lab"][nm]], np.int64)
            first = np.empty(len(code), np.int64)  # C5: copies of a stored draw share its key (left out together)
            for k, l in enumerate(labs):
                sel = code == k
                first[sel] = self._first_index(l)[D["idx"][nm][sel]]
            keys.append(code * 10 ** 7 + first)
        T = L["rows"][L["xrow"][0]]
        X = [T["lh"][sp]]
        gp = None
        if goal is not None:
            kb = np.concatenate([np.full(N, GOAL_BODIES_POOL), np.full(n_outer, GOAL_BODIES_OUTER)])
            gp = self._eig_goal_success(goal, L, D, kb, rng)
            X.append(gp["p"][sp])
        X = np.stack(X, 1)
        nm_ = self._eig_nmc(obs, YO, sp, [k[so] for k in keys], [k[sp] for k in keys], D["logw"][sp], H, X)
        ess = np.exp(2 * nm_["logSall"] - _lse(nm_["logS2"][:, cols["magnetic"]], 1))
        lo_, hi_ = (math.log(v) for v in EIG_ESS_POOL)
        lam = np.clip((np.log(np.maximum(ess, 1e-300)) - lo_) / (hi_ - lo_), 0.0, 1.0)
        (ZP, nq), (ZO, _) = self._eig_summaries(obs, YP), self._eig_summaries(obs, YO)
        fP, fO = DP["fold"], DO["fold"]
        res = {"name": cand.get("name", ""), "measurements": cand["measurements"], "cost": cand.get("cost"),
               "n_outer": n_outer, "n_inner_per_magnetic_mechanism": int(N // n_mag), "n_inner": N,
               "eig_indicators": {}}
        ar = np.arange(n_outer)
        wpool = np.exp(DP["logw"] - DP["logw"].max())
        for nm in names:
            labs = self._eig_labels(nm)
            w = np.array([self.w[nm][l] for l in labs], float)
            if nm in cond:  # marginal of a conditional indicator as the draws generate it (sum over draws of p(label))
                w = np.array([wpool[DP["lab"][nm] == l].sum() for l in labs]) + 1e-12
            w = w / w.sum()
            HM = float(-(w * np.log(w)).sum())
            y = np.array([labs.index(l) for l in DO["lab"][nm]])
            r = {"labels": labs, "prior_weights": dict(zip(labs, w.tolist())), "prior_entropy_nats": HM}
            if zero[nm]:
                r.update(stored_draw_bracket=[0.0, 0.0], stored_draw_se=0.0, stored_draw_closed=True, stored_draw_nats=0.0)
                r.update(lower_bound_source="structural zero", lower_pooled=0.0)
                r.update(nats=0.0, value_se=0.0, value_kind="structural zero", nested_estimate=0.0,
                         nested_estimate_se=0.0, lower=0.0, upper=0.0, se=0.0, lower_bound_halves=[0.0, 0.0],
                         lower_bound_stable=True,
                         bracket_closed=True, converged=True, structural_zero=True, lower_nested=0.0,
                         lower_classifier=0.0,
                         expected_posterior_weights=dict(zip(labs, w.tolist())),
                         p_identify_map=float(np.mean(np.argmax(w) == y)), p_identify_map_bracket=[
                             float(np.mean(np.argmax(w) == y))] * 2, n_nonfinite_outer=0)
                res["eig_indicators"][nm] = r
                continue
            lS = nm_["logS"][:, cols[nm]]
            with np.errstate(divide="ignore", invalid="ignore"):
                q_nmc = np.exp(lS - nm_["logSall"][:, None])
            q_nmc = np.where(np.isfinite(q_nmc), q_nmc, 0.0)
            q_clf = np.empty((n_outer, len(labs)))
            sw = np.exp(DP["logw"] - DP["logw_own"][nm])
            yP = np.array([labs.index(l) for l in DP["lab"][nm]])
            for f in (0, 1):
                tr, ev = fP == f, fO != f
                if not ev.any():
                    continue
                Tt = np.eye(len(labs))[yP[tr]]
                clf = _Softmax(ZP[tr], Tt, sw[tr], nq)
                prior_tr = np.maximum((Tt * sw[tr][:, None]).sum(0) / sw[tr].sum(), 1e-12)
                lp = clf.log_prob(ZO[ev]) + np.log(w)[None] - np.log(prior_tr)[None]
                q_clf[ev] = np.exp(lp - _lse(lp, 1)[:, None])
            qp = (1 - EIG_EPS) * (lam[:, None] * q_nmc + (1 - lam[:, None]) * q_clf) + EIG_EPS * w[None]
            qn = (1 - EIG_EPS) * q_nmc + EIG_EPS * w[None]
            qc = (1 - EIG_EPS) * q_clf + EIG_EPS * w[None]
            # three valid Barber-Agakov bounds on the same simulated experiments (pooled; exact nested posterior with
            # the generating stored draw left out, i.e. v3.1's exclude-own estimate made leave-out; classifier); the
            # lower end is the largest of them
            cands = {"pooled": qp, "nested": qn, "classifier": qc}
            lqs = {k: np.log(v[ar, y]) for k, v in cands.items()}
            bounds = {k: HM + float(v.mean()) for k, v in lqs.items()}
            src = max(bounds, key=bounds.get)
            q, lq, lower = cands[src], lqs[src], bounds[src]
            se = float(lq.std(ddof=1) / math.sqrt(n_outer))
            # upper: generating draw included in its own stratum (log p(y | m) with own term, v3.1 style)
            logp = lS - np.log(N * w)[None]
            logp[ar, y] = np.logaddexp(lS[ar, y], own) - np.log(N * w[y] + 1)
            lpi = logp + np.log(w)[None]
            q_in = np.exp(lpi - _lse(lpi, 1)[:, None])
            upper = HM + float(np.mean(np.log(np.maximum(q_in[ar, y], 1e-300))))
            # stored-draw estimate (v3.1 style): inner set including copies of the generating stored draw
            lSf = np.logaddexp(lS, nm_["logS_copies"][:, cols[nm]])
            lq_sd = np.clip(lSf[ar, y] - np.logaddexp(nm_["logSall"], nm_["logSall_copies"]), -50.0, 0.0)
            sd_lower, sd_se = HM + float(lq_sd.mean()), float(lq_sd.std(ddof=1) / math.sqrt(n_outer))
            sd_closed = bool(abs(upper - sd_lower) <= max(0.05, 2 * sd_se))
            halves, stable = _eig_halves(lq, fO, HM)
            closed = bool(abs(upper - lower) <= max(0.05, 2 * se))
            # C5: a closed bracket whose upper estimate lies below the lower bound (Monte Carlo noise where the EIG is
            # near zero) is flagged; the value and bracket stay exactly as C4 computes them
            inverted = bool(closed and lower > upper)
            # closed bracket: the posterior is accurate, so report the nested estimate in its lower-variance form
            # H(prior) - E[H(q(. | y))] (the generating label averaged out); otherwise the lower bound
            ent = -(qn * np.log(qn)).sum(1)  # the exact nested posterior (generating stored draw left out)
            nested_val, nested_se = HM - float(ent.mean()), float(ent.std(ddof=1) / math.sqrt(n_outer))
            clipped = bool(closed and not (lower <= nested_val <= upper))
            r.update(nats=min(max(nested_val, lower), upper) if closed else max(lower, 0.0),
                     value_se=nested_se if closed else se,
                     value_kind=("nested estimate (bracket closed)" + (", clipped to the bracket" if clipped else ""))
                     if closed else "lower bound", value_clipped_to_bracket=clipped,
                     nested_estimate=nested_val, nested_estimate_se=nested_se,
                     lower=lower, upper=upper, se=se, lower_bound_source=src, lower_bound_halves=halves,
                     lower_bound_stable=stable, bracket_closed=closed,
                     converged=stable, structural_zero=False,
                     stored_draw_bracket=[sd_lower, upper], stored_draw_se=sd_se, stored_draw_closed=sd_closed,
                     stored_draw_nats=0.5 * (max(sd_lower, 0.0) + upper) if sd_closed else None,
                     lower_pooled=bounds["pooled"], lower_nested=bounds["nested"],
                     lower_classifier=bounds["classifier"],
                     expected_posterior_weights=dict(zip(labs, q.mean(0).tolist())),
                     p_identify_map=float(np.mean(np.argmax(q, 1) == y)),
                     p_identify_map_bracket=[float(np.mean(np.argmax(q, 1) == y)),
                                             float(np.mean(np.argmax(q_in, 1) == y))],
                     n_nonfinite_outer=int((~np.isfinite(lq)).sum()))
            if inverted:  # additive keys only
                r["bracket_inverted"] = True
                r["bracket_note"] = ("closed bracket with its upper estimate below its lower bound: Monte Carlo noise; "
                                     "the estimate is indistinguishable from zero")
            res["eig_indicators"][nm] = r
        for nm, r in res["eig_indicators"].items():  # flat keys (v3.1 names for magnetic and density)
            k = self._eig_key(nm)
            res[k + "_nats"], res[k + "_bracket"] = r["nats"], [r["lower"], r["upper"]]
            res[k + "_se"], res[k + "_converged"] = r["value_se"], r["converged"]
            res[k + "_lower_bound"], res[k + "_lower_bound_se"] = r["lower"], r["se"]
            res[k + "_lower_bound_stable"], res[k + "_bracket_closed"] = r["lower_bound_stable"], r["bracket_closed"]
            res[k + "_value_se"] = r["value_se"]
            if r.get("bracket_inverted"):  # C5: additive flag (see bracket_note)
                res[k + "_bracket_inverted"] = True
            if nm in ("magnetic", "density"):
                res[f"n_nonfinite_outer_{nm}"] = r["n_nonfinite_outer"]
                res[f"expected_posterior_{nm}_weights"] = r["expected_posterior_weights"]
                res[f"p_identify_{nm}_mechanism_map"] = r["p_identify_map"]
                res[f"p_identify_{nm}_mechanism_bracket"] = r["p_identify_map_bracket"]
        # all parameters: prior-contrastive bound with the generating draw included (a lower bound)
        c = own - (np.logaddexp(own, nm_["logSall"]) - math.log(N + 1))
        res["eig_all_parameters_nats"] = float(c.mean())
        res["eig_all_parameters_se"] = float(c.std(ddof=1) / math.sqrt(n_outer))
        # target route: expected posterior sd of the route-level ln Hci (an upper bound: held-out prediction error)
        wP = np.exp(DP["logw"] - DP["logw"].max())
        wP /= wP.sum()
        tP, tO = T["lh"][sp], T["lh"][so]
        spec_var = float(wP @ T["sig_spec"][sp] ** 2)
        pred = np.empty(n_outer)
        for f in (0, 1):
            tr, ev = fP == f, fO != f
            if ev.any():
                pred[ev] = _ridge(ZP[tr], tP[tr], wP[tr], ZO[ev], nq)
        mean_nmc = np.where(np.isfinite(nm_["mean_x"][:, 0]), nm_["mean_x"][:, 0], pred)
        pred = lam * mean_nmc + (1 - lam) * pred
        res["target_route"] = tgt
        res["ln_Hci_sd_target_now"] = math.sqrt(float(wP @ tP ** 2 - (wP @ tP) ** 2) + spec_var)
        res["ln_Hci_sd_target_expected_after"] = math.sqrt(float(np.mean((tO - pred) ** 2)) + spec_var)
        res["ln_Hci_sd_reduction_target"] = res["ln_Hci_sd_target_now"] - res["ln_Hci_sd_target_expected_after"]
        res["ess_inner_median"] = float(np.median(ess))
        res["ess_inner_q10"] = float(np.quantile(ess, 0.10))
        res["pooling_weight_nested_mean"] = float(lam.mean())
        if goal is not None:
            res["goal"] = self._eig_goal_summary(goal, gp, nm_["mean_x"][:, 1], lam, ZP, ZO, nq, fP, fO, DP, sp, so)
            res["eig_goal_event_nats"] = res["goal"]["eig_event_nats"]
            res["goal_p_success_sd_reduction"] = (res["goal"]["p_success_epistemic_sd_now"]
                                                  - res["goal"]["p_success_epistemic_sd_expected_after"])
        return res

    # ---------------------------------------------------------------- goal-oriented target (v4)
    def _eig_goal_parse(self, goal):
        if not isinstance(goal, dict) or "route" not in goal or "spec" not in goal:
            raise ValueError('goal needs {"route": {...}, "spec": {...}, "rho_observed": optional}')
        bad = sorted(set(goal) - {"route", "spec", "rho_observed"})
        if bad:
            raise ValueError(f"unknown goal keys {bad}; allowed: route, spec, rho_observed")
        spec = dict(_eig_spec_check(dict(goal["spec"])))
        route = dict(goal["route"])
        if self._has_c7() and route.get("anneal", "none") != "none":
            raise ValueError("an eig goal at an annealed route would score C5 Br without the anneal layer (C7.1); use "
                             "joint_target() for the goal probability and anneal_eig() for anneal experiments")
        if "T_operating_C" in spec:
            route["T_measure_C"] = float(spec["T_operating_C"])
        self._route(route)
        ro = goal.get("rho_observed")
        if ro is not None and not (_is_number(ro) and ro > 0):
            raise ValueError("goal rho_observed must be a positive number (g/cm3 or relative)")
        return {"route": route, "spec": spec, "rho_observed": ro}

    def _eig_goal_success(self, goal, L, D, kb, rng):
        """p(goal | draw): the fraction of kb[i] simulated bodies made by the goal route that meet the spec, scored as
        in p_meets_spec (measured values: body density scatter, specimen, piece, Br, squareness and curve-shape scatter;
        the temperature scenario at T_operating_C), with the draw's parameters and its shared latents (preparation
        effect, degas state, H0 direction, conditional indicators) held fixed. This is the aleatoric success
        probability given the draw; its spread over draws is the epistemic uncertainty."""
        xi = L["xrow"][-1]
        R, r = L["rows"][xi], dict(L["routes"][xi])
        A = self._arrays(r)
        rep = np.repeat(np.arange(D["n"]), kb)
        nb = len(rep)
        if goal["rho_observed"] is not None:
            ro = float(goal["rho_observed"])
            rho = (ro / RHO_TD if ro > 1.5 else ro) + 0.005 * rng.standard_normal(nb)
        else:
            rho = np.clip(R["mu_rho"][rep] + R["sig_rho"][rep] * rng.standard_t(5.0, nb), 0.3, 1.02)
        nu = L["nuis"]
        eps = nu["eps"][r["prep"]]
        kw = {"indicators": {k: v[rep] for k, v in L["kw"]["indicators"].items()}} if L["kw"] else {}
        u = R["sig_spec"][rep] * rng.standard_normal(nb)
        lat = self._magnetic_latent(A, r, D["lab"]["magnetic"][rep], D["idx"]["magnetic"][rep], rho,
                                    nu["degas"][r["prep"]][rep], nu["z_hull"][rep], u_spec=u,
                                    eps=None if eps is None else eps[rep], **kw)
        hci = np.exp(lat["lh"]) * np.exp(lat["sig_piece"] * rng.standard_normal(nb))
        br = np.exp(lat["lbr"] + lat["sig_Br"] * rng.standard_normal(nb))
        sq = 1.0 / (1.0 + np.exp(-(lat["sq"] + lat["sig_sq"] * rng.standard_normal(nb))))
        hci_T, br_T = self._temperature(lat, hci, br, float(r["T_measure_C"]), rng)
        bh = self._bh(br_T, hci_T, sq, rng)
        vals = {"Hci_kOe_min": hci_T, "BHmax_MGOe_min": bh, "Br_kGs_min": br_T, "density_rel_min": rho}
        ok = np.ones(nb, bool)
        for k, v in vals.items():
            if k in goal["spec"]:
                ok &= np.asarray(v) >= float(goal["spec"][k])
        return {"p": np.bincount(rep, weights=ok.astype(float), minlength=D["n"]) / np.asarray(kb, float),
                "kb": np.asarray(kb)}

    def _eig_goal_summary(self, goal, gp, p_nmc, lam, ZP, ZO, nq, fP, fO, DP, sp, so):
        """Information about reaching the goal. G = 1 if the next body made by the goal route meets the spec.
        theta = a joint draw (parameters and shared latents). p(theta) = P(G = 1 | theta) is aleatoric (body-to-body
        and measurement scatter); its spread over the posterior is epistemic and is what an experiment can reduce."""
        pP, pO = gp["p"][sp], gp["p"][so]
        kP, kO = float(GOAL_BODIES_POOL), float(GOAL_BODIES_OUTER)
        wP = np.exp(DP["logw"] - DP["logw"].max())
        wP /= wP.sum()
        pbar = float(wP @ pP)
        res = {"route": self._route(goal["route"]), "spec": goal["spec"], "rho_observed": goal["rho_observed"],
               "p_success_now": pbar,
               "p_success_now_se": float(math.sqrt(max(wP @ (pP - pbar) ** 2, 1e-30) / max(1.0 / (wP @ wP), 1.0)))}
        var_now = max(float(wP @ (pP - pbar) ** 2 - wP @ (pP * (1 - pP)) / (kP - 1)), 0.0)
        H_event = float(_bern_entropy(pbar))
        h_ale = _bern_entropy(pO) + np.where((pO > 0) & (pO < 1), 1.0 / (2 * kO), 0.0)
        aleatoric = float(min(np.mean(np.minimum(h_ale, math.log(2))), H_event))
        if pbar <= 1e-12 or pbar >= 1 - 1e-12:
            res.update(no_draw_meets_spec=bool(pbar <= 1e-12), all_draws_meet_spec=bool(pbar >= 1 - 1e-12),
                       event_entropy_nats=0.0, aleatoric_entropy_nats=0.0, epistemic_entropy_nats=0.0,
                       eig_event_nats=0.0, eig_event_bracket=[0.0, 0.0], eig_event_se=0.0, eig_event_converged=True,
                       eig_event_lower_bound_stable=True, eig_event_bracket_closed=True,
                       p_success_epistemic_sd_now=0.0, p_success_epistemic_sd_expected_after=0.0,
                       expected_posterior_p_success_quantiles={f"q{int(100 * q):02d}": pbar for q in GOAL_QUANTILES})
            return res
        pred = np.empty(len(pO))
        for f in (0, 1):
            tr, ev = fP == f, fO != f
            if ev.any():
                Tt = np.stack([1 - pP[tr], pP[tr]], 1)
                clf = _Softmax(ZP[tr], Tt, wP[tr], nq)
                pred[ev] = np.exp(clf.log_prob(ZO[ev])[:, 1])
        p_nmc = np.where(np.isfinite(p_nmc), p_nmc, pred)
        m = (1 - EIG_EPS) * (lam * p_nmc + (1 - lam) * pred) + EIG_EPS * pbar
        m = np.clip(m, 1e-12, 1 - 1e-12)
        c = pO * np.log(m) + (1 - pO) * np.log1p(-m)
        lower = H_event + float(c.mean())
        se = float(c.std(ddof=1) / math.sqrt(len(c)))
        upper = max(H_event - aleatoric, 0.0)
        halves, stable = _eig_halves(c, fO, H_event)
        after = max(float(np.mean((pO - m) ** 2) - np.mean(pO * (1 - pO)) / (kO - 1)), 0.0)
        res.update(no_draw_meets_spec=False, all_draws_meet_spec=False,
                   event_entropy_nats=H_event, aleatoric_entropy_nats=aleatoric,
                   epistemic_entropy_nats=upper,
                   eig_event_nats=max(lower, 0.0), eig_event_bracket=[lower, upper], eig_event_se=se,
                   eig_event_lower_bound_halves=halves, eig_event_lower_bound_stable=stable,
                   eig_event_converged=stable, eig_event_bracket_closed=bool(upper - lower <= max(0.05, 2 * se)),
                   p_success_epistemic_sd_now=math.sqrt(var_now),
                   p_success_epistemic_sd_expected_after=min(math.sqrt(after), math.sqrt(var_now)),
                   expected_posterior_p_success_quantiles={f"q{int(round(100 * q)):02d}": float(np.quantile(m, q))
                                                           for q in GOAL_QUANTILES},
                   expected_posterior_p_success_mean=float(m.mean()),
                   bodies_per_draw={"inner": int(kP), "outer": int(kO)},
                   definitions="G = 1 if the next body made by the goal route meets every spec key (scored on measured "
                               "values as p_meets_spec). theta = a joint posterior draw (parameters and shared latents); "
                               "p(theta) = P(G = 1 | theta). event_entropy = H(E[p]); aleatoric = E[H(p(theta))] "
                               "(scatter no experiment removes); epistemic = event - aleatoric = I(G; theta), the most "
                               "any experiment can gain about G. eig_event: I(G; y) bracketed by a Barber-Agakov lower "
                               "bound and I(G; theta). p_success_epistemic_sd: sd of p(theta) now and the expected "
                               "posterior sd after the experiment (an upper bound: held-out prediction error). "
                               "expected_posterior_p_success_quantiles: the distribution over simulated outcomes of the "
                               "updated success probability (approximate; shrunk toward the mean where the approximation "
                               "is weak)")
        return res

    # ---------------------------------------------------------------- conditioning on new data
    @staticmethod
    def _sem_observation(o):
        """C5: validated SEM/BSE content of one observation (one piece): {"grain": [(stage, um), ...], "bse": [(stage,
        "spread"|"pooled"), ...]} (empty dict when it carries none). A number takes o["stage"] (required for a grain
        size; a BSE class defaults to as sintered); a dict gives {stage: value} for both stages of the piece."""
        def stage(s):
            s = STAGE_ALIASES.get(s, s)
            if s not in SEM_STAGES:
                raise ValueError(f"stage must be one of {SEM_STAGES} (pass 'stage' or a dict {{stage: value}})")
            return s
        out = {}
        if "grain_size_sem_um" in o:
            g = o["grain_size_sem_um"]
            pairs = [(stage(s), v) for s, v in g.items()] if isinstance(g, dict) else [(stage(o.get("stage")), g)]
            for _, v in pairs:
                if not (_is_number(v) and float(v) > 0):
                    raise ValueError("grain_size_sem_um must be a positive number (um)")
            out["grain"] = [(s, float(v)) for s, v in pairs]
        if "gb_phase_bse" in o:
            c = o["gb_phase_bse"]
            pairs = [(stage(s), v) for s, v in c.items()] if isinstance(c, dict) else \
                [(stage(o.get("stage", "as_sintered")), c)]
            for _, v in pairs:
                if v not in BSE_CLASSES:
                    raise ValueError(f"gb_phase_bse must be one of {BSE_CLASSES}")
            bse = [(s, v) for s, v in pairs if v != "unreadable"]
            if bse:
                out["bse"] = bse
        return out

    def _sem_loglik(self, so, lat):
        """Log-likelihood of one observation's SEM/BSE content under each draw (lat from _magnetic_latent(sem=True))."""
        ll = np.zeros(len(lat["W_bse"]))
        for st, v in so.get("grain", []):
            ll += _n_logpdf(math.log(v), np.log(lat["g_" + st]) + np.log(lat["c5_cm"]), GRAIN_FIELD_LOG_SD)
        for st, c in so.get("bse", []):
            W = self.fw.bse_stage_continuity({"c5_rann": lat["c5_rann"]}, lat["W_bse"], st, np)
            z = (W - lat["c5_Wstar"]) / lat["c5_tau"]
            ll += np.log(np.maximum(_std_normal_cdf(z if c == "spread" else -z), 1e-300))
        return ll

    def _copies(self, m):
        """For the stored draws of mechanism m: the group index among identical stored draws (a resampled posterior
        holds copies), the number of distinct draws, and the index of each draw's first stored copy (the draw itself
        when m has no copies)."""
        cache = self.__dict__.setdefault("_copies_cache", {})
        if m not in cache:
            src = self.post[m]
            M = np.stack([np.asarray(src[k], float) for k in sorted(src)], 1)
            if m in self.ltp:
                M = np.concatenate([M, self.ltp[m]], 1)
            _, first, inv = np.unique(M, axis=0, return_index=True, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            cache[m] = (inv, int(inv.max()) + 1, np.asarray(first)[inv])
        return cache[m]

    def _distinct(self, m):
        inv, nd, _ = self._copies(m)
        return inv, nd

    def _first_index(self, m):
        return self._copies(m)[2]

    def condition(self, observations, k_latent=32, seed=12345):
        """Importance-reweight the stored draws by new observations; returns (updated model, info).
        observations: list of {"route": {...}, "rho_observed": optional, "Hci_kOe", "Br_kGs", "Hk_kOe",
        "density_rel", "grain_um" (each optional)}; each observation is one new body (its own specimen effect).
        Shared latents are integrated numerically with k_latent samples per draw: new-preparation effects (shared by
        observations naming the same prep), the degas state per preparation (route["degas"] occurrence probability
        times route["degas_completeness"]) and H0's discrepancy direction. Mechanism weights are updated by each mechanism's marginal likelihood.
        C5 observables (declared measurement models, README "SEM and BSE outputs"): "grain_size_sem_um" (um) with
        "stage" ("as_sintered" or "after_900_520"), or a dict {stage: um}; "gb_phase_bse": "spread", "pooled" or
        "unreadable" (adds nothing) at "stage" (default as sintered), or a dict {stage: class}. They need no density.
        info["ess_per_stratum"] and info["ess_per_mechanism_distinct"] count distinct stored draws; ess_per_mechanism
        counts stored draws, copies included."""
        rng = np.random.default_rng(seed)
        logw = {m: np.zeros(len(v["sig_rho" if m.startswith("D") else "sig_spec"])) for m, v in self.post.items()}
        obs = [dict(o, route=self._route(o["route"])) for o in observations]
        sem_obs = {i: self._sem_observation(o) for i, o in enumerate(obs)}
        dc_by_prep = {}
        for o in obs:
            key = json.dumps(o["route"]["degas_completeness"], sort_keys=True)
            if dc_by_prep.setdefault(o["route"]["prep"], key) != key:
                raise ValueError(f"observations sharing preparation {o['route']['prep']!r} must share degas_completeness")
        fd_by_prep = {}
        for o in obs:
            _check_prep_feedstock(fd_by_prep, o["route"], "observations")
        dens_obs = [o for o in obs if "density_rel" in o]
        # C7: the stored SPS density draws describe as-sintered bodies and C5's Br at an annealed stage is scaled by the
        # anneal layer, which this importance reweighting does not integrate; refuse rather than mis-assign the stage
        staged = [i for i, o in enumerate(obs) if o["route"]["anneal"] != "none" and ("density_rel" in o or "Br_kGs" in o)]
        if staged:
            raise ValueError(f"observations {staged}: density_rel or Br_kGs at an annealed stage cannot condition the "
                             "as-sintered SPS density draws (C7: annealing densifies; the anneal layer is calibrated "
                             "separately, build/c7_anneal_fit.py). Pass the as-sintered reading with anneal 'none', or "
                             "refit owner-side")
        mag_keys = ("Hci_kOe", "Br_kGs", "Hk_kOe", "grain_um")
        mag_obs = [o for i, o in enumerate(obs) if any(k in o for k in mag_keys) or sem_obs[i]]
        sem_of = {id(o): sem_obs[i] for i, o in enumerate(obs)}
        preps = sorted({o["route"]["prep"] for o in obs})
        K = k_latent
        if dens_obs:
            for m in self._mechs("density"):
                Kd = len(self.post[m]["lnk"])
                zs = {p: rng.standard_normal((Kd, K)) for p in preps}
                acc = np.zeros((Kd, K))
                p = {k: v[:, None] for k, v in self.post[m].items()}
                for o in dens_obs:
                    y = float(o["density_rel"])
                    y = y / RHO_TD if y > 1.5 else y
                    A = self._arrays(o["route"])
                    if o["route"]["prep"] in self.preps:
                        off = p["tau_prep"] * p[f"z_prep_{self.preps.index(o['route']['prep'])}"]
                        mu = self._dens_mu(p, m, A, off, o["route"])[:, 0]
                        acc += _t_logpdf(y, mu, p["sig_rho"][:, 0])[:, None]
                    else:
                        for k in range(K):
                            off = p["tau_prep"] * zs[o["route"]["prep"]][:, k:k + 1]
                            mu = self._dens_mu(p, m, A, off, o["route"])[:, 0]
                            acc[:, k] += _t_logpdf(y, mu, p["sig_rho"][:, 0])
                logw[m] += _lme(acc, 1)
        new_ltp = {}
        if mag_obs:
            for o in mag_obs:  # backlog item 2: no silent 0.95 TD default for an unknown density
                if any(k in o for k in mag_keys) and o.get("rho_observed", o.get("density_rel")) is None:
                    raise ValueError("a magnetic observation needs the body's measured density: pass rho_observed "
                                     "(g/cm3 or relative) or density_rel")
            for m in self._mechs("magnetic"):
                Km = len(self.post[m]["sig_spec"])
                idx = np.arange(Km)
                lab = np.array([m] * Km)
                levels = LT_LEVELS if m in self.ltp else ("load",)
                acc = {L: np.zeros((Km, K)) for L in levels}
                zh = rng.standard_normal((Km, K)) if m == "H0" else np.zeros((Km, K))
                dz = {p: rng.random((Km, K)) for p in preps}
                nondefault = [o for o in mag_obs if not self._completeness_is_default(o["route"])
                              and o["route"]["degas_completeness"] != "complete"]
                dstate, deps = {}, {}
                if nondefault:
                    extra = rng.spawn(1)[0]
                    for pr in preps:
                        ro = next((o["route"] for o in mag_obs if o["route"]["prep"] == pr), None)
                        if ro is not None and not self._completeness_is_default(ro) \
                                and ro["degas_completeness"] != "complete":
                            dstate[pr] = self._sample_completeness(ro["degas_completeness"], (Km, K), extra)
                            deps[pr] = extra.uniform(0.0, 1.0, (Km, K))
                for o in mag_obs:
                    r = o["route"]
                    A = self._arrays(r)
                    so = sem_of[id(o)]
                    rho_v = o.get("rho_observed", o.get("density_rel"))
                    if rho_v is None:  # C5 SEM/BSE-only observation: grain and W do not depend on the density
                        rho_v = E_RHO_PLACEHOLDER
                    rho_v = float(rho_v) / RHO_TD if float(rho_v) > 1.5 else float(rho_v)
                    rho = np.full(Km, rho_v)
                    shared = m == "H0" or 0.0 < float(r["degas"]) < 1.0 or r["prep"] in dstate
                    r = dict(r, _degas_expected=self._expected_state(r))
                    for k in (range(K) if shared else [0]):
                        degas_ind = (dz[r["prep"]][:, k] < float(r["degas"])).astype(float)
                        e = None
                        if r["prep"] in dstate:
                            degas_ind = degas_ind * dstate[r["prep"]][:, k]
                            e = deps[r["prep"]][:, k]
                        for L in levels:
                            ind = {"load_timing": np.full(Km, L)} if m in self.ltp else None
                            lat = self._magnetic_latent(A, r, lab, idx, rho, degas_ind, zh[:, k], eps=e, indicators=ind,
                                                        sem=bool(so))
                            ll = np.zeros(Km)
                            if so:
                                ll += self._sem_loglik(so, lat)
                            if "Hci_kOe" in o:
                                ll += _n_logpdf(math.log(o["Hci_kOe"]), lat["lh"],
                                                np.sqrt(lat["sig_spec"] ** 2 + lat["sig_piece"] ** 2))
                            if "Br_kGs" in o:
                                ll += _n_logpdf(math.log(o["Br_kGs"]), lat["lbr"], lat["sig_Br"])
                            if "Hk_kOe" in o and "Hci_kOe" in o and 0 < o["Hk_kOe"] < o["Hci_kOe"]:
                                ratio = o["Hk_kOe"] / o["Hci_kOe"]
                                ll += _n_logpdf(math.log(ratio / (1 - ratio)), lat["sq"], lat["sig_sq"])
                            if "grain_um" in o:
                                ll += _n_logpdf(math.log(o["grain_um"]), lat["lgrain"], 0.15)
                            if shared:
                                acc[L][:, k] += ll
                            else:
                                acc[L] += ll[:, None]
                lik = np.stack([_lme(acc[L], 1) for L in levels], 1)          # (Km, n_levels) log p(y_new | theta, LT)
                if m in self.ltp:
                    lp = np.log(np.maximum(self.ltp[m], 1e-300)) + lik
                    tot = _lse(lp, 1)
                    new_ltp[m] = np.exp(lp - tot[:, None])
                    logw[m] += tot
                else:
                    logw[m] += lik[:, 0]
        new_dw, newW, ess = {}, {}, {}
        for kind in ("density", "magnetic"):
            lm = {}
            for m in self._mechs(kind):
                base = np.log(self.dw[m] / self.dw[m].sum())
                tot = _lse(base + logw[m])
                lm[m] = tot
                wv = np.exp(base + logw[m] - tot)
                new_dw[m] = wv
                ess[m] = float(1.0 / np.sum(wv ** 2))
            z = _lse(np.array([math.log(max(self.w[kind][m], 1e-300)) + lm[m] for m in self._mechs(kind)]))
            newW[kind] = {m: math.exp(math.log(max(self.w[kind][m], 1e-300)) + lm[m] - z) for m in self._mechs(kind)}
        ltp = {m: new_ltp.get(m, v) for m, v in self.ltp.items()}
        if ltp:
            tot = np.zeros(len(LT_LEVELS))
            for m in self._mechs("magnetic"):
                tot += newW["magnetic"][m] * (new_dw[m] @ ltp[m])
            newW["load_timing"] = {L: float(x) for L, x in zip(LT_LEVELS, tot / tot.sum())}
        for k in self.w:
            newW.setdefault(k, dict(self.w[k]))
        clone = object.__new__(PhysicsModel)
        clone.__dict__.update(self.__dict__)
        clone.dw, clone.w, clone.ltp = new_dw, newW, ltp
        info = {"mechanism_weights": newW, "ess_per_mechanism": ess,
                "warning": "ESS below ~200 for a mechanism that still carries weight means the stored draws no longer "
                           "represent its posterior; refit owner-side (NumPyro) instead. Shared latents of a new "
                           "preparation are integrated for the reweighting but redrawn in later predictions."}
        # C5: effective sample sizes on distinct stored draws (a resampled posterior stores copies), per mechanism
        # and per (coercivity mechanism, load-timing explanation) stratum
        ess_d, strata, n_distinct = {}, {}, {}
        for kind in ("density", "magnetic"):
            for m in self._mechs(kind):
                inv, nd = self._distinct(m)
                n_distinct[m] = nd
                g = np.bincount(inv, weights=new_dw[m], minlength=nd)
                ess_d[m] = float(g.sum() ** 2 / np.sum(g ** 2))
                if kind == "magnetic":
                    P = ltp[m] if m in ltp else np.ones((len(new_dw[m]), 1))
                    labs = LT_LEVELS if m in ltp else ("load",)
                    strata[m] = {}
                    for j, L in enumerate(labs):
                        gv = np.bincount(inv, weights=new_dw[m] * P[:, j], minlength=nd)
                        strata[m][L] = float(gv.sum() ** 2 / np.sum(gv ** 2)) if gv.sum() > 0 else 0.0
        info["ess_per_mechanism_distinct"] = ess_d
        info["ess_per_stratum"] = strata
        info["distinct_stored_draws"] = n_distinct
        info["ess_note"] = ("ess_per_mechanism counts stored draws, copies included; ess_per_mechanism_distinct and "
                            "ess_per_stratum count distinct stored draws (copies merged)")
        clone.conditioning = {"observations": observations, **info}
        return clone, info


_DEFAULT = None


def _default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load()
    return _DEFAULT


def predict(route, **kw):
    return _default().predict(route, **kw)


def p_meets_spec(route, spec, **kw):
    return _default().p_meets_spec(route, spec, **kw)


def eig(candidates, **kw):
    return _default().eig(candidates, **kw)


def forecast_record(route, **kw):
    return _default().forecast_record(route, **kw)


def sps_log(route, **kw):
    return _default().sps_log(route, **kw)


def coarse_number_fraction(route, **kw):
    return _default().coarse_number_fraction(route, **kw)


def joint_target(route, **kw):
    return _default().joint_target(route, **kw)


def anneal_eig(candidates, **kw):
    return _default().anneal_eig(candidates, **kw)


def condition_anneal(observations, **kw):
    return _default().condition_anneal(observations, **kw)
