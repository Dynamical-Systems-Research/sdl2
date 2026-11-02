"""C7 anneal-layer calibration on the one paired anneal return (#8, P5) (owner side; numpy only).

Prior draws of the anneal and density-reading parameters (PRIORS, each with its basis) are weighted by the likelihood of
#8's two reported readings, per mechanism, with the reading nuisances (method of each reading, piece pairing, piece
effect, the body's true as-sintered density) integrated by Monte Carlo inside each prior draw. The as-sintered level of
#8 enters the SPS density update separately (build/c7_density_update.py); here it only pins the start of the anneal
(a cut posterior, as C6's microstructure calibration). Stored per mechanism: N_STORE resampled posterior draws and
N_STORE prior draws. Reported: mechanism evidence and posterior weights, per declared #8-hold scenario and per reading
method combination (conditional alternatives), the dose each mechanism needs, and posterior predictions for retained
lower-density pieces under the same program, with the information a retained-piece anneal would carry about the
mechanism.

Run (from the package): python3 -B build/c7_anneal_fit.py .   (writes anneal_posterior.npz and build/ANNEAL_FIT_REPORT.json;
deterministic for SEED)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG))   # the package's anneal.py
import anneal as an  # noqa: E402

SEED = 20261023
N_PRIOR = 200_000        # prior draws per mechanism
N_NUIS = 16              # nuisance samples (reading methods, pairing, piece effects, true start) per prior draw
N_STORE = 4000
RHO_TD = 7.62
OBS = {"body": "#8", "route": "850 C SPS (RPI panel route; #8 SPS log not released)",
       "source_id": "gmail:1a15becba0000b94", "specimen_id": "RPI-P1-850-02",
       "reading_ids": {"before": "RPI-850-02-DENSITY-01", "after": "RPI-850-02-DENSITY-AFTER-FIRST-ANNEAL-01"},
       "as_sintered_g_cm3": 7.07, "after_900_g_cm3": 7.58, "reported_pct_TD": {"as_sintered": 92, "after_900": 99},
       "source": "Ziyi Wu (RPI) Gmail 1a15becba0000b94, 2026-10-20 20:46 New York; released P5",
       "not_stated": ["density method (either stage)", "uncertainty", "TD denominator", "anneal hold, atmosphere, "
                      "heating and cooling", "whether one piece was measured at both stages"],
       "rounding_g_cm3": 0.005}
HOLD_SCENARIOS_H = (1.0, 2.0, 4.0)
HOLD_PRIOR = (0.25, 0.50, 0.25)   # declared: 2 h is the September timeline's planned program, not confirmed for #8

PRIORS = {
    "c7_lnk900": dict(spec=("uniform", math.log(0.01), math.log(30.0)), units="ln(1/h)",
                      meaning="ln of the densification rate constant at 900 C (per hour of hold)",
                      basis="declared, wide: 2 h doses from 0.02 (no visible change) to 60 (complete); #8 identifies "
                            "only rate x hold, per mechanism"),
    "c7_Q": dict(spec=("uniform", 150.0, 400.0), units="kJ/mol",
                 meaning="apparent activation energy of the anneal densification",
                 basis="declared: liquid-assisted solution-reprecipitation and grain-boundary diffusion ranges; no "
                       "record has an anneal at a second temperature, so it only matters away from 900 C"),
    "c7_Tliq": dict(spec=("uniform", 630.0, 690.0), units="C",
                    meaning="onset of the Nd-rich liquid (rate switches on above it)",
                    basis="Nd-Fe-B ternary eutectic about 655-665 C (the investigators' shared reference value 655 C); "
                          "+/-30 C for composition and oxygen"),
    "c7_wliq": dict(spec=("uniform", 8.0, 20.0), units="C", meaning="width of the liquid onset",
                    basis="declared"),
    "c7_phifloor": dict(spec=("uniform", 0.0, 0.008), units="-",
                        meaning="residual porosity a load-free anneal cannot remove (gas-filled, trapped)",
                        basis="declared: 0-0.8 % of the full density"),
    "c7_phic": dict(spec=("uniform", 0.05, 0.12), units="-",
                    meaning="porosity at which half the pores are closed (CLOSED mechanism; also the open-porosity "
                            "share in the Archimedes reading)",
                    basis="pore closure in sintering is usually reported at 88-95 %TD; declared 5-12 % porosity"),
    "c7_wc": dict(spec=("uniform", 0.008, 0.02), units="-", meaning="width of the open-to-closed transition",
                  basis="declared"),
    "c7_ropen": dict(spec=("loguniform", 0.005, 0.15), units="-",
                     meaning="relative rate of open porosity (CLOSED mechanism)",
                     basis="declared: open compacts coarsen rather than shrink without load; 0.5-15 % of the closed "
                           "rate (Opus v2's +0.04 RD for open bodies lies inside this range at #8's dose)"),
    "c7_cap": dict(spec=("uniform", 0.03, 0.15), units="-",
                   meaning="largest porosity one anneal can remove (CAPACITY mechanism)",
                   basis="declared: the Nd-rich liquid volume (roughly 3-6 %) and its redistribution bound pore filling; "
                         "Qwen v1.6's +0.067 relative increment lies inside"),
    "c7_rhofull": dict(spec=("uniform", 0.995, 1.010), units="fraction of 7.62",
                       meaning="full (theoretical) density of this alloy relative to the tool's 7.62 basis",
                       basis="RPI's 92 %/99 % imply 7.64-7.70 if rounded to the nearest percent, 7.60-7.66 if "
                             "truncated; earlier panel %TD imply 7.59-7.67; uniform 7.58-7.70"),
    "c7_hold8_h": dict(spec=("choice", HOLD_SCENARIOS_H, HOLD_PRIOR), units="h",
                       meaning="the hold #8 actually had at 900 C (needed only to translate #8's dose to an explicit "
                               "program; predictions 'as #8' do not use it)",
                       basis="declared scenario weights: 2 h planned (September timeline 900 + 520 C / 2 h each), "
                             "1 h and 4 h alternatives; RPI has not stated it"),
    "c7_kopen": dict(spec=("uniform", 0.0, 1.0), units="-",
                     meaning="share of open porosity an Archimedes reading counts as solid",
                     basis="declared: 0 for a sealed or saturated-weight (dry/wet/suspended) protocol, 1 for an "
                           "uncorrected immersion; RPI's protocol is not stated"),
    "c7_sd_arch": dict(spec=("uniform", 0.002, 0.006), units="fraction of 7.62",
                       meaning="sd of one Archimedes reading", basis="declared: 0.015-0.045 g/cm3"),
    "c7_bgeo": dict(spec=("normal", -0.01, 0.015), units="fraction",
                    meaning="relative bias of a mass/dimension (geometric) reading",
                    basis="declared: rough faces, chipped edges and graphite foil overstate the volume of a small "
                          "disk (reads low); the panel's reported values and cylinder arithmetic disagree by up to "
                          "16 % (#3) and 9 % (#5), so the method of the reported values is itself unclear"),
    "c7_sd_geo": dict(spec=("uniform", 0.005, 0.02), units="fraction of 7.62",
                      meaning="sd of one geometric reading", basis="declared: 0.04-0.15 g/cm3"),
    "c7_sdpiece": dict(spec=("uniform", 0.002, 0.02), units="fraction of 7.62",
                       meaning="sd of a piece's density about its body's (SPS radial/axial gradients)",
                       basis="declared: 0.015-0.15 g/cm3; no piece map has been released"),
}
MECH_PRIOR = {m: 0.25 for m in an.ANNEAL_MECHS}
# nuisances of the #8 readings, integrated inside each draw
P_SAME_PIECE = 0.5                # declared: piece pairing not confirmed
P_ARCHIMEDES = {"as_sintered": 0.5, "after_900": 0.5}   # declared: method not stated for either stage
START_PRIOR = (0.86, 0.99)        # true as-sintered relative density of #8's body (uniform; broad on purpose)


def sample_prior(rng, n):
    out = {}
    for k, v in PRIORS.items():
        kind, *a = v["spec"]
        if kind == "uniform":
            out[k] = rng.uniform(a[0], a[1], n)
        elif kind == "loguniform":
            out[k] = np.exp(rng.uniform(math.log(a[0]), math.log(a[1]), n))
        elif kind == "normal":
            out[k] = rng.normal(a[0], a[1], n)
        elif kind == "choice":
            out[k] = rng.choice(np.asarray(a[0], float), size=n, p=np.asarray(a[1], float))
        else:
            raise ValueError(kind)
    return out


def _tile(p, k):
    return {key: np.repeat(v, k) for key, v in p.items()}


def loglik_pair(p, mech, rng, nuis=N_NUIS, cond=None):
    """log p(#8 readings | draw) with the nuisances integrated by Monte Carlo (nuis samples per draw). cond fixes
    some nuisances: {"methods": (m_as, m_an), "same_piece": bool}."""
    n = len(p["c7_lnk900"])
    q = _tile(p, nuis)
    m = np.repeat(np.asarray(mech).astype(str), nuis) if np.ndim(mech) else np.full(n * nuis, str(mech))
    N = n * nuis
    rho_body = rng.uniform(START_PRIOR[0], START_PRIOR[1], N)
    same = (rng.random(N) < P_SAME_PIECE) if not cond or "same_piece" not in cond else np.full(N, cond["same_piece"])
    eta_a = q["c7_sdpiece"] * rng.standard_normal(N)
    eta_b = np.where(same, eta_a, q["c7_sdpiece"] * rng.standard_normal(N))
    piece_a = an.piece_density(q, m, rho_body, eta_a, [])
    piece_b1 = an.piece_density(q, m, rho_body, eta_b, [(900.0, q["c7_hold8_h"])])
    y_as, y_an = OBS["as_sintered_g_cm3"] / RHO_TD, OBS["after_900_g_cm3"] / RHO_TD
    ll = np.zeros(N)
    for stage, y, piece in (("as_sintered", y_as, piece_a), ("after_900", y_an, piece_b1)):
        if cond and "methods" in cond:
            meth = cond["methods"][0 if stage == "as_sintered" else 1]
            ll += an.reading_logpdf(q, y, piece, meth)
        else:
            pa = P_ARCHIMEDES[stage]
            la = an.reading_logpdf(q, y, piece, "archimedes") + math.log(pa)
            lg = an.reading_logpdf(q, y, piece, "geometric") + math.log(1 - pa)
            mx = np.maximum(la, lg)
            ll += mx + np.log(np.exp(la - mx) + np.exp(lg - mx))
    ll = ll.reshape(n, nuis)
    mx = ll.max(1, keepdims=True)
    return (mx[:, 0] + np.log(np.mean(np.exp(ll - mx), 1)))


def lse(a):
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def fit(rng):
    res = {}
    for mech in an.ANNEAL_MECHS:
        p = sample_prior(rng, N_PRIOR)
        ll = loglik_pair(p, mech, rng)
        logZ = lse(ll) - math.log(N_PRIOR)
        w = np.exp(ll - ll.max())
        w /= w.sum()
        ess = float(1.0 / np.sum(w ** 2))
        res[mech] = dict(p=p, ll=ll, w=w, logZ=logZ, ess=ess)
    return res


def mech_posterior(res, sel=None):
    """Posterior mechanism weights from the evidences (sel: boolean mask per mechanism over the prior draws, e.g. one
    hold scenario; evidence is then the mean likelihood within the mask)."""
    lz = {}
    for m, r in res.items():
        ll = r["ll"] if sel is None else r["ll"][sel(r["p"])]
        lz[m] = lse(ll) - math.log(len(ll)) if len(ll) else -np.inf
    z = lse(np.array([math.log(MECH_PRIOR[m]) + lz[m] for m in lz]))
    return {m: math.exp(math.log(MECH_PRIOR[m]) + lz[m] - z) for m in lz}, lz


def resample(r, rng, n=N_STORE):
    idx = rng.choice(len(r["w"]), size=n, p=r["w"])
    return {k: v[idx] for k, v in r["p"].items()}


def qsum(x):
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q95": float(q[4])}


RETAINED = {  # released as-sintered readings (g/cm3) of bodies that could be annealed; method as recorded
    "#1 (750 C)": (4.83, "reported (method not stated; cylinder arithmetic 4.97)"),
    "#4 (800 C)": (5.77, "reported (method not stated; cylinder arithmetic 5.73)"),
    "#6 (800 C)": (5.63, "reported (method not stated; cylinder arithmetic 5.64)"),
    "#7 (850 C)": (7.01, "Archimedes (corrected from the 7.10 typo)"),
}


def predict_retained(post, rng, n=N_STORE):
    """Posterior predictive of the annealed piece's TRUE density and of a same-piece Archimedes reading (sealed or not, as
    the draw's c7_kopen) after the same program as #8, starting from the released reading taken as the true start
    (the reported value's own method uncertainty is added as a 1.5 % relative sd)."""
    out = {}
    for name, (y, basis) in RETAINED.items():
        row = {"as_sintered_g_cm3": y, "basis": basis}
        for mech, p in post.items():
            rho0 = np.clip(y / RHO_TD * (1.0 + 0.015 * rng.standard_normal(n)), 0.3, 1.0)
            rho1 = an.anneal_density_as8(p, np.full(n, mech), rho0)
            read = an.reading_mean(p, rho1, "archimedes") + an.reading_sd(p, "archimedes") * rng.standard_normal(n)
            row[mech] = {"true_g_cm3": {k: v * RHO_TD for k, v in qsum(rho1).items()},
                         "archimedes_reading_g_cm3": {k: v * RHO_TD for k, v in qsum(read).items()},
                         "gain_g_cm3_median": float(np.median(rho1 - rho0) * RHO_TD)}
        out[name] = row
    return out


def info_about_mechanism(post, weights, rng, body_g_cm3, n=N_STORE, bins=None):
    """Mutual information (nats) between the mechanism label and a same-piece before/after density gain of one retained
    body (both readings by the same stated method, sealed Archimedes, piece effect cancels), estimated by histogram
    of the gain under each mechanism's posterior predictive. Also P(mechanism | gain in each bin)."""
    bins = np.linspace(-0.15, 2.8, 60) if bins is None else bins
    H = {}
    for mech, p in post.items():
        rho0 = np.clip(body_g_cm3 / RHO_TD * (1.0 + 0.015 * rng.standard_normal(n)), 0.3, 1.0)
        rho1 = an.anneal_density_as8(p, np.full(n, mech), rho0)
        sd = an.reading_sd(p, "archimedes")
        gain = (rho1 - rho0) * RHO_TD + math.sqrt(2.0) * sd * RHO_TD * rng.standard_normal(n)
        h, _ = np.histogram(np.clip(gain, bins[0], bins[-1] - 1e-9), bins=bins)
        H[mech] = (h + 0.5) / (h.sum() + 0.5 * len(h))
    w = np.array([weights[m] for m in post])
    P = np.stack([H[m] for m in post])            # (M, B) p(gain bin | mech)
    marg = w @ P
    mi = float(np.sum(w[:, None] * P * (np.log(P) - np.log(marg[None, :]))))
    ent = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))
    return {"mutual_information_nats": mi, "mechanism_entropy_nats": ent, "share_resolved": mi / ent if ent else 0.0}


SENSITIVITY = {  # alternative nuisance priors for the artifact (NULL) explanation; reported, not stored
    "wide_piece_gradient": {"c7_sdpiece": ("uniform", 0.002, 0.05)},
    "wide_geometric_bias": {"c7_bgeo": ("normal", -0.02, 0.03), "c7_sd_geo": ("uniform", 0.005, 0.03)},
    "both_wide": {"c7_sdpiece": ("uniform", 0.002, 0.05), "c7_bgeo": ("normal", -0.02, 0.03),
                  "c7_sd_geo": ("uniform", 0.005, 0.03)},
}


def sensitivity(seed, n=60_000):
    out = {}
    for name, change in SENSITIVITY.items():
        saved = {k: PRIORS[k]["spec"] for k in change}
        try:
            for k, spec in change.items():
                PRIORS[k]["spec"] = spec
            rng = np.random.default_rng([seed, 23, len(out)])
            lz = {}
            for mech in an.ANNEAL_MECHS:
                p = sample_prior(rng, n)
                ll = loglik_pair(p, mech, rng)
                lz[mech] = lse(ll) - math.log(n)
            z = lse(np.array([math.log(MECH_PRIOR[m]) + lz[m] for m in lz]))
            out[name] = {"change": {k: list(v) for k, v in change.items()},
                         "mechanism_weights": {m: math.exp(math.log(MECH_PRIOR[m]) + lz[m] - z) for m in lz}}
        finally:
            for k, spec in saved.items():
                PRIORS[k]["spec"] = spec
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = fit(rng)
    weights, logZ = mech_posterior(res)
    by_hold = {f"{h:g}h": mech_posterior(res, sel=lambda p, h=h: p["c7_hold8_h"] == h)[0] for h in HOLD_SCENARIOS_H}
    # conditional alternatives on the reading methods and the pairing (fixed nuisances, same prior draws per mechanism)
    alt = {}
    rng_alt = np.random.default_rng([SEED, 7])
    for meths in (("archimedes", "archimedes"), ("geometric", "archimedes"), ("archimedes", "geometric"),
                  ("geometric", "geometric")):
        for same in (True, False):
            lz = {}
            for mech in an.ANNEAL_MECHS:
                p = {k: v[:40_000] for k, v in res[mech]["p"].items()}
                ll = loglik_pair(p, mech, rng_alt, cond={"methods": meths, "same_piece": same})
                lz[mech] = lse(ll) - math.log(len(ll))
            z = lse(np.array([math.log(MECH_PRIOR[m]) + lz[m] for m in lz]))
            alt[f"{meths[0]}->{meths[1]}, {'same piece' if same else 'different pieces'}"] = {
                m: math.exp(math.log(MECH_PRIOR[m]) + lz[m] - z) for m in lz}
    rng_s = np.random.default_rng([SEED, 11])
    post = {m: resample(res[m], rng_s) for m in an.ANNEAL_MECHS}
    prior = {m: {k: v[:N_STORE] for k, v in res[m]["p"].items()} for m in an.ANNEAL_MECHS}
    # stored package file
    arrays = {}
    for m in an.ANNEAL_MECHS:
        for k in an.ANNEAL_PARAMS + an.MEASURE_PARAMS:
            arrays[f"post__{m}__{k}"] = post[m][k].astype(np.float64)
            arrays[f"prior__{m}__{k}"] = prior[m][k].astype(np.float64)
    dose = {m: qsum(an.rate(post[m], 900.0) * post[m]["c7_hold8_h"]) for m in an.ANNEAL_MECHS}
    meta = {"layer": "C7 anneal densification and density reading", "version": "c7-anneal-1.2",
            "observations": [OBS], "priors": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                                  for kk, vv in v.items()} for k, v in PRIORS.items()},
            "mechanism_prior": MECH_PRIOR, "mechanism_weights": weights, "logZ": logZ,
            "ess": {m: res[m]["ess"] for m in res}, "n_prior_per_mechanism": N_PRIOR, "n_nuisance": N_NUIS,
            "n_store": N_STORE, "seed": SEED, "mech_sources": an.MECH_SOURCES,
            "nuisance_priors": {"same_piece": P_SAME_PIECE, "archimedes": P_ARCHIMEDES,
                                "true_as_sintered_start": list(START_PRIOR)},
            "by_hold_scenario": by_hold, "conditional_on_methods_and_pairing": alt,
            "dose_k900_x_hold": dose,
            "status": "calibrated on one paired return (#8); mechanism weights are prior-dominated where the pair "
                      "cannot separate them (see mechanism_weights vs mechanism_prior)"}
    np.savez(out / "anneal_posterior.npz", __meta__=np.array(json.dumps(meta)), **arrays)
    (out / "build").mkdir(exist_ok=True)
    pred = predict_retained(post, np.random.default_rng([SEED, 13]))
    info = {name: info_about_mechanism(post, weights, np.random.default_rng([SEED, 17, i]), y)
            for i, (name, (y, _)) in enumerate(RETAINED.items())}
    report = {"mechanism_weights": weights, "mechanism_prior": MECH_PRIOR, "logZ": logZ,
              "ess": {m: res[m]["ess"] for m in res}, "by_hold_scenario": by_hold,
              "conditional_on_methods_and_pairing": alt, "dose_k900_x_hold": dose,
              "posterior_parameters": {m: {k: qsum(post[m][k]) for k in ("c7_lnk900", "c7_phic", "c7_cap",
                                                                           "c7_ropen", "c7_rhofull", "c7_kopen",
                                                                           "c7_sdpiece")} for m in post},
              "retained_piece_predictions_same_program_as_8": pred,
              "information_from_retained_piece_anneal": info,
              "nuisance_prior_sensitivity": sensitivity(SEED),
              "package_file_sha256": hashlib.sha256((out / "anneal_posterior.npz").read_bytes()).hexdigest()}
    (out / "build" / "ANNEAL_FIT_REPORT.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1])
    print(json.dumps({k: rep[k] for k in ("mechanism_weights", "ess", "by_hold_scenario")}, indent=1))
    print(json.dumps(rep["conditional_on_methods_and_pairing"], indent=1))
    for b, row in rep["retained_piece_predictions_same_program_as_8"].items():
        print(b, {m: (round(row[m]["true_g_cm3"]["q05"], 2), round(row[m]["true_g_cm3"]["median"], 2),
                      round(row[m]["true_g_cm3"]["q95"], 2)) for m in ("LPS", "CLOSED", "CAPACITY", "NULL")})
    print(json.dumps(rep["information_from_retained_piece_anneal"], indent=1))
