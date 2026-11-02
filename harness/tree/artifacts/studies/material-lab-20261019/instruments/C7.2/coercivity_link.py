"""C6 coercivity link (work package 1): microstructure state -> simulated Hci, Br, Hk/Hci, (BH)max distributions.

Pure numpy, vectorized, no file I/O. Owner-built model; every output is SIMULATED and UNSCORED in this campaign window.

Physics (see README_BLOCK.md for the full statement):
  Hci = alpha * mu0 H_N - N_eff(D) * Js, with mu0 H_N = 2 K1 mu0 / Js and N_eff = n ln(D / delta_B)   (Kronmueller form)
  alpha = alpha_E * R_alpha(c, J) / R_alpha(c_E, J) + defect_offset [+ anneal term]
    R_alpha(c, J): alpha-frame coverage response of a grain-network model (>= 10^3 grains) calibrated on 48 magnum.np
                   polycrystal loops (Phase 2a; validated on Fischbacher 2017's 40 nm cube), embedded below as tables
                   derived from phase2/runs/network_cal.json (sha256 SOURCE_SHA256) by derive_tables.py.
    alpha_E:       pinned per draw so that historical sample E (Hci 7.5 kOe; RPI measurement, pieces 7.28-7.71) is
                   reproduced at E's route and E's measured microstructure (the absolute anchor).
    J:             boundary-film magnetization, never drawn per world: every call is computed at J in {0, 0.6, 1.2} T
                   and marginalized with JGB_POSTERIOR inside the call.
    c:             face coverage by any film; c >= c_SEM (etched-section lower bound) when measurable; otherwise a
                   declared log-normal prior on the coverage deficit 1 - c driven by BSE placement (pocket share,
                   faint-line coverage), anchored at D and E and widened away from them.
  Chemistry (v2): Hci is multiplied by exp(ln_hci_chem_offset), the tool's per-draw C5 chemistry and porosity offset
    relative to the D/E reference state (oxidation and milled-powder hydrogen of fine feedstock, degassing completeness,
    open porosity). It is exactly 0 at historical D's and E's routes, so the anchors and the E hold-out do not move.
  Br = Js * rho * <cos theta> * (1 - f_Nd) * (1 + kappa) * R(Hci);  (BH)max from the parametric second-quadrant curve,
  capped at Br^2 / 4 mu0.
"""
from __future__ import annotations

import numpy as np

LINK_VERSION = "c6-coercivity-link-2"
STATUS = ("simulated; unscored in this window; depends on the boundary film's magnetization, which no measurement in "
          "this window can reveal")
# D/E basis: Phase 2a rejection calibration (phase2/code/forecast7_phase2.py) starting from the prior {0: 0.2, 0.6: 0.5,
# 1.2: 0.3} T; a 1.2 T film can rarely reproduce RPI's D/E contrast (21 % of draws), so its weight falls.
_JW = np.array([0.262, 0.654, 0.083])
JGB_POSTERIOR = {0.0: float(_JW[0] / _JW.sum()), 0.6: float(_JW[1] / _JW.sum()), 1.2: float(_JW[2] / _JW.sum())}
JGB_LEVELS = (0.0, 0.6, 1.2)

# ---- embedded tables (derive_tables.py on phase2/runs/network_cal.json) --------------------------------------------
SOURCE_SHA256 = "777338094aaffa8d788ce00586053d4fa8bf1a9467d08fd6f57aae1e4fd541fa"
H_REF_SIM_T = 2.627829   # mean simulated Hci, non-magnetic film, full coverage, 47 nm grains (4 seeds)
X_SIM_T = 0.674357       # N_eff(47 nm) * Js = 0.17 * 1.61 T * ln(47/4)
C_GRID = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0])
R_HCI = {0.0: np.array([0.4, 0.402, 0.418, 0.491, 0.632, 0.798, 0.911, 1.0]),
         0.6: np.array([0.4, 0.377, 0.368, 0.403, 0.496, 0.593, 0.663, 0.719]),
         1.2: np.array([0.4, 0.348, 0.308, 0.306, 0.338, 0.369, 0.399, 0.42])}
R_ALPHA = {J: (R_HCI[J] * H_REF_SIM_T + X_SIM_T) / (H_REF_SIM_T + X_SIM_T) for J in JGB_LEVELS}
FIT_RATIO_N20 = {"c0.0_J0.0": 0.412, "c0.0_J0.6": 0.412, "c0.0_J1.2": 0.412, "c0.7_J0.0": 0.508, "c0.7_J0.6": 0.43,
                 "c0.7_J1.2": 0.331, "c0.9_J0.0": 0.748, "c0.9_J0.6": 0.574, "c0.9_J1.2": 0.367, "c1.0_J0.0": 1.0,
                 "c1.0_J0.6": 0.72, "c1.0_J1.2": 0.42}
MICROMAG_TARGET_RATIO = {"c0.7_J0.0": 0.589, "c0.0_J0.0": 0.422, "c0.9_J0.6": 0.632, "c0.0_J1.2": 0.422,
                         "c1.0_J1.2": 0.419, "c1.0_J0.6": 0.643, "c0.9_J1.2": 0.394, "c0.0_J0.6": 0.422,
                         "c0.7_J1.2": 0.342, "c0.7_J0.6": 0.465, "c1.0_J0.0": 1.0, "c0.9_J0.0": 0.815}

# ---- anchors (RPI measurements and owner metrology; agent readings of RPI images) ----------------------------------
E_HCI_KOE = 7.50                 # mean of E_1-E_3 (7.28, 7.52, 7.71) [RPI]
E_HCI_LOGSD = 0.03
E_POCKET_SHARE, E_FAINT = 0.7645, 0.047   # metrology bse_phase_by_body.csv / faint_lines_baseline.json (T10, l2.5)
E_ND_RICH = 0.0376                # Otsu
E_GRAIN_MEDIAN_UM, E_GRAIN_LOGSD, E_GRAIN_SIGMA = 3.0, 0.20, 0.48   # metrology: E900 intercept -> 2.7-3.3 um in 3D
D_POCKET_SHARE = 0.985            # D anchor (faint lines unreadable: blurred 20 kV JPEG)
DELTA_E = 0.015                   # E's median coverage deficit (c_E ~ 0.985; Phase 2a: c_E ~ U(0.97, 1))
K_POCKET = 8.4                    # ln-deficit per unit pocket share: calibrated so D's marginal Hci median = 2.86 kOe (c_D ~0.90)
M_FAINT = 10.0                    # ln-deficit per unit faint-line coverage: DECLARED, uncalibrated
S_BASE, S_HULL, S_NAN = 0.35, 0.5, 0.3   # log-sd of the deficit prior: at anchors, per hull unit, if faint unreadable
CAL_GRAIN_UM = (1.5, 4.0)         # grain range treated as in-domain (records 1.5-3.5 um); outside -> flag
ANNEAL_LOWT_GAIN = (0.25, 0.30)   # Hci gain of a 520-600 C post-sinter anneal, commercial sintered (Li 2021 STAM s2)
ALIGN_PENALTY_AT_092 = (0.82, 0.87)   # along-axis Hci factor at <cos> 0.92 (Matsuura 2018; summary-level)
HR_KOE, KR = 0.15, 2.0            # remanence retention R(H) = 1 / (1 + (HR/H)^KR) (declared; C: Hci 0.21 -> R ~0.66)
_MU0 = 4e-7 * np.pi
_HGRID = np.linspace(0.0, 1.0, 161)


def _ralpha(c, J):
    return np.interp(c, C_GRID, R_ALPHA[J])


def _intrinsic(rng, n):
    K1 = rng.normal(4.3, 0.15, n) * 1e6
    Js = rng.normal(1.61, 0.01, n)
    HN = 2 * K1 * _MU0 / Js * 10.0          # kOe
    nB = rng.normal(0.17, 0.03, n)
    dB = rng.normal(4.0, 0.4, n) * 1e-3     # um
    return HN, Js * 10.0, nB, dB             # Js in kG


def _deficit_mean_and_sd(pocket, faint):
    """Declared placement -> coverage-deficit prior (log-normal), anchored at E and D; widened off the D-E line."""
    pocket = np.asarray(pocket, float); faint = np.asarray(faint, float)
    fnan = ~np.isfinite(faint)
    pk = np.where(np.isfinite(pocket), pocket, E_POCKET_SHARE)
    ln_mu = np.log(DELTA_E) + K_POCKET * (pk - E_POCKET_SHARE) - np.where(fnan, 0.0, M_FAINT * (np.nan_to_num(faint) - E_FAINT))
    span = D_POCKET_SHARE - E_POCKET_SHARE
    hull = np.maximum(0.0, E_POCKET_SHARE - pk) / span + np.maximum(0.0, pk - D_POCKET_SHARE) / span
    faint_line = E_FAINT * (D_POCKET_SHARE - pk) / span          # faint coverage along the D-E line at this pocket share
    hull = hull + np.where(fnan, 0.0, np.abs(np.nan_to_num(faint) - faint_line) / E_FAINT)
    hull = hull + np.where(np.isfinite(pocket), 0.0, 1.0)
    sd = S_BASE + S_HULL * hull + np.where(fnan, S_NAN, 0.0)
    return ln_mu, sd, hull


def _coverage(rng, n, pocket, faint, c_sem):
    ln_mu, sd, hull = _deficit_mean_and_sd(pocket, faint)
    delta = np.exp(ln_mu + sd * rng.standard_normal(n))
    c = 1.0 - np.minimum(delta, 1.0)
    c_sem = np.asarray(c_sem, float)
    c = np.where(np.isfinite(c_sem), np.maximum(c, c_sem), c)
    return np.clip(c, 0.0, 1.0), hull


def _bhmax(br, hci, hk):
    q = np.log(0.1) / np.log(np.clip(hk, 1e-3, 0.999))
    h = _HGRID[None, :] * hci[:, None]
    J = br[:, None] * (1.0 - (_HGRID[None, :]) ** q[:, None])
    bh = h * np.clip(J - h, 0.0, None)
    return np.minimum(bh.max(axis=1), br ** 2 / 4.0)


def _bcast(x, n, default):
    if x is None:
        return np.full(n, default, float)
    a = np.asarray(x, float)
    return np.full(n, float(a), float) if a.ndim == 0 else a


def link(state: dict, rng: np.random.Generator) -> dict:
    """state: arrays of length n from one tool draw (see README_BLOCK.md). Returns per-output {"marginal", "by_jgb"},
    "flags" and "status". J_GB is marginalized inside the call; a world never carries a J_GB draw."""
    n = len(np.atleast_1d(state["grain_median_um"]))
    D = _bcast(state["grain_median_um"], n, 3.0)
    sig = _bcast(state.get("grain_sigma_ln"), n, 0.48)
    rho = _bcast(state.get("density_rel"), n, 0.974)
    fnd = _bcast(state.get("nd_rich_area_fraction"), n, E_ND_RICH)
    stage = np.broadcast_to(np.asarray(state.get("stage", "as_sintered")), (n,))
    annealed = stage != "as_sintered"
    pocket = _bcast(state.get("pocket_share"), n, np.nan)
    faint = _bcast(state.get("faint_line_coverage"), n, np.nan)
    pocket_as = state.get("pocket_share_as_sintered"); faint_as = state.get("faint_line_coverage_as_sintered")
    used_annealed_placement = np.zeros(n, bool)
    if pocket_as is not None:
        pa = _bcast(pocket_as, n, np.nan); fa = _bcast(faint_as, n, np.nan)
        pocket = np.where(annealed, pa, pocket); faint = np.where(annealed, fa, faint)
    else:
        used_annealed_placement = annealed.copy()
    c_sem = _bcast(state.get("coverage_sem_lower"), n, np.nan)
    cos = np.clip(_bcast(state.get("alignment_cos_theta"), n, 0.5), 0.5, 0.95)
    doff = np.nan_to_num(_bcast(state.get("defect_offset"), n, 0.0))
    chem = np.nan_to_num(_bcast(state.get("ln_hci_chem_offset"), n, 0.0))

    HN, JsK, nB, dB = _intrinsic(rng, n)
    # E anchor: alpha_E per draw from E's Hci at E's grain (mean of its log-normal distribution)
    DE = np.exp(rng.normal(np.log(E_GRAIN_MEDIAN_UM), E_GRAIN_LOGSD, n)) * np.exp(E_GRAIN_SIGMA ** 2 / 2)
    aE = (E_HCI_KOE * np.exp(rng.normal(0, E_HCI_LOGSD, n)) + nB * np.log(DE / dB) * JsK) / HN
    cE, _ = _coverage(rng, n, np.full(n, E_POCKET_SHARE), np.full(n, E_FAINT), np.full(n, np.nan))
    c, hull = _coverage(rng, n, pocket, faint, c_sem)
    # anneal: load-free re-wetting (900 C is above the 580-606 C melting of metallic Nd) for both annealed stages;
    # the 520 C step adds the literature low-temperature-anneal gain (a declared alpha increment, not fitted to E)
    r = rng.beta(1.5, 1.5, n)
    c_eff = np.where(annealed, c + r * (1.0 - c), c)
    gain = rng.uniform(*ANNEAL_LOWT_GAIN, n)
    Deff = D * np.exp(sig ** 2 / 2)
    neff = nB * np.log(Deff / dB) * JsK
    u_al = rng.uniform(*ALIGN_PENALTY_AT_092, n)
    align_pen = 1.0 - (1.0 - u_al) * (cos - 0.5) / 0.42
    kappa = rng.normal(0.0, 0.06, n)
    hk_iso = rng.uniform(0.22, 0.31, n)
    hk_al = rng.uniform(0.80, 0.95, n)
    hk = hk_iso + (hk_al - hk_iso) * (cos - 0.5) / 0.45

    out = {k: {"by_jgb": {}} for k in ("hci_kOe", "br_kGs", "hk_over_hci", "bhmax_MGOe")}
    clipped_any = np.zeros(n, bool)
    for J in JGB_LEVELS:
        a = aE * _ralpha(c_eff, J) / _ralpha(cE, J) + doff
        h_as = aE * _ralpha(c, J) / _ralpha(cE, J) + doff
        h_as = h_as * HN - neff
        a = np.where(stage == "after_900_520", a + gain * np.maximum(h_as, 0.0) / HN, a)
        hraw = a * HN - neff
        clipped = hraw < 0.05
        clipped_any |= clipped
        hci = np.maximum(hraw, 0.05) * np.exp(chem) * align_pen
        R = 1.0 / (1.0 + (HR_KOE / hci) ** KR)
        br = JsK * rho * cos * (1.0 - fnd) * (1.0 + kappa) * R
        bh = _bhmax(br, hci, hk)
        out["hci_kOe"]["by_jgb"][J] = hci
        out["br_kGs"]["by_jgb"][J] = br
        out["hk_over_hci"]["by_jgb"][J] = hk.copy()
        out["bhmax_MGOe"]["by_jgb"][J] = bh
    # marginal over the J_GB posterior: per draw, pick the J level with the posterior weights (a mixture sample)
    pick = rng.choice(len(JGB_LEVELS), size=n, p=np.array([JGB_POSTERIOR[J] for J in JGB_LEVELS]))
    for k in out:
        stack = np.stack([out[k]["by_jgb"][J] for J in JGB_LEVELS])
        out[k]["marginal"] = stack[pick, np.arange(n)]
    out["flags"] = {
        "clipped_breakdown": clipped_any,
        "grain_extrapolated": (Deff < CAL_GRAIN_UM[0]) | (Deff > CAL_GRAIN_UM[1]),
        "alignment_extrapolated": cos > 0.5,
        "out_of_anchor_hull": hull > 0.5,
        "anneal_placement_used": used_annealed_placement,
        "anneal_term_declared": annealed,
        "chemistry_coupled": np.abs(chem) > 1e-12,
    }
    out["status"] = STATUS
    out["version"] = LINK_VERSION
    return out


def inverse_map_cell(D_um: float, c: float, J_target: float, rng: np.random.Generator, n: int = 100000,
                     e_grain_median_um: float = 3.5, aligned: bool = False) -> dict:
    """The COERCIVITY_LINK_SCOPE section-3 computation (target film J_target, E's own film from the posterior), using the
    embedded tables. Default E grain 3.5 um as in that scope; the link itself uses the metrology update (3.0 um)."""
    HN, JsK, nB, dB = _intrinsic(rng, n)
    DE = np.exp(rng.normal(np.log(e_grain_median_um), 0.25, n))
    aE = (7.5 * np.exp(rng.normal(0, 0.03, n)) + nB * np.log(DE / dB) * JsK) / HN
    JE = rng.choice(JGB_LEVELS, n, p=np.array([JGB_POSTERIOR[J] for J in JGB_LEVELS]))
    cE = rng.uniform(0.97, 1.0, n)
    RE = np.select([JE == J for J in JGB_LEVELS], [_ralpha(cE, J) for J in JGB_LEVELS])
    h = aE * _ralpha(np.full(n, c), J_target) / RE * HN - nB * np.log(D_um / dB) * JsK
    if aligned:
        h = h * rng.uniform(0.82, 0.87, n)
    return {"median": float(np.median(h)), "P20": float(np.mean(h >= 20))}

# ---------------------------------------------------------------------------------------------------------------------
# Tool adapter (campaign session, 2026-10-18). Everything above this line is work package 1's module byte for byte
# (sha256 1c2906b4b7ddb7db77da6675e7327d4bd51e2cbd23da16378126f2d25fcac170). The physics tool calls link(state, rng) once
# with its nested per-draw state (C6 tool LINK_STATE_KEYS); this adapter calls the flat single-stage link once per stage
# and returns quantile summaries. A flat state (work package 1's interface) still goes straight to the flat link.
_link_flat = link
CORE_SHA256 = "1c2906b4b7ddb7db77da6675e7327d4bd51e2cbd23da16378126f2d25fcac170"
# C7 (2026-10-23): the adapter takes each stage's own body density (state["rho_rel_by_stage"], from the tool's anneal
# layer) for the stage's Br; without it every stage reads the as-sintered density, as in C6. The core above is unchanged.
ADAPTER_VERSION = "c7-link-adapter-1"
_OUT_NAMES = {"hci_kOe": "hci_link_kOe", "br_kGs": "br_link_kGs", "hk_over_hci": "hk_over_hci_link",
              "bhmax_MGOe": "bhmax_link_MGOe"}


def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(x.mean()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "n": int(x.size)}


def link(state: dict, rng: np.random.Generator) -> dict:
    """Tool entry point. Nested tool state (keys 'stages' and 'nd_rich') -> per-stage flat calls and summaries;
    a flat state -> work package 1's flat link unchanged."""
    if not ("stages" in state and "nd_rich" in state):
        return _link_flat(state, rng)
    n = int(state["n"])
    nd_as = state["nd_rich"]["as_sintered"]
    stages = {}
    for s in state["stages"]:
        nd = state["nd_rich"][s]
        flat = {"grain_median_um": np.asarray(state["grain_model_um"][s], float),
                "grain_sigma_ln": np.asarray(state["grain_sigma_ln"], float),
                "density_rel": np.asarray((state.get("rho_rel_by_stage") or {}).get(s, state["rho_rel"]), float),
                "nd_rich_area_fraction": np.asarray(nd["nd_rich_area_fraction"], float),
                "pocket_share": np.asarray(nd["pocket_share"], float),
                "faint_line_coverage": np.asarray(nd["faint_line_coverage"], float),
                "pocket_share_as_sintered": np.asarray(nd_as["pocket_share"], float),
                "faint_line_coverage_as_sintered": np.asarray(nd_as["faint_line_coverage"], float),
                "coverage_sem_lower": np.full(n, np.nan),   # no etched section in the tool state
                "stage": np.full(n, s),
                "alignment_cos_theta": np.full(n, float(state["alignment_cos_theta"])),
                "ln_hci_chem_offset": np.asarray(state.get("ln_hci_chem_offset", np.zeros(n)), float)}
        out = _link_flat(flat, rng)
        st = {_OUT_NAMES[k]: {"marginal": _q(v["marginal"]),
                              "by_jgb": {f"{J:g}": _q(a) for J, a in v["by_jgb"].items()}}
              for k, v in out.items() if k in _OUT_NAMES}
        st["flags"] = {k: float(np.mean(v)) for k, v in out["flags"].items()}
        stages[s] = st
    return {"status": STATUS, "version": LINK_VERSION, "core_sha256": CORE_SHA256, "adapter": ADAPTER_VERSION,
            "stage_density": ("each stage's own body density (C7 anneal layer)" if state.get("rho_rel_by_stage")
                              else "the as-sintered body density at every stage (C6)"),
            "jgb_posterior_T": {f"{J:g}": w for J, w in JGB_POSTERIOR.items()},
            "flags_note": "flag values are the share of draws flagged",
            "stages": stages}
