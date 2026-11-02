"""C7 SPS density update on the as-sintered readings released after the C5/C6 density fit (cutoff P3): #7 (7.01 g/cm3,
Archimedes; the 7.10 typo is superseded) and #8 (7.07 g/cm3 as sintered, method not stated). Owner side; numpy only.

Method: the C6 package's own importance reweighting of its stored density draws (PhysicsModel.condition's density
branch, re-implemented here so the #8 reading can carry its method uncertainty): per stored draw, the likelihood of the
two readings with the 850 C bodies' shared new-preparation effect integrated by K_LATENT samples, Student-t(5) with the
draw's sig_rho (the C5 density likelihood), widened for #8 by SD_METHOD_8 (method not stated). #8's after-900 reading is
not used here (it is the anneal layer's observation). Outputs: per-draw weights (stored with the C7 posterior as dw__*),
updated density-mechanism weights, effective sample sizes on distinct draws, and the predictive at the 850 C route
before and after. Alternatives reported: separate preparations for #7 and #8; #8 alone; #7 alone.

Run (from the C7 package): python3 -B build/c7_density_update.py ../C6-development .   (writes posterior_tool.npz,
build/DENSITY_UPDATE_REPORT.json; deterministic for SEED)
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
SEED = 20261023
K_LATENT = 128
SD_METHOD_8 = 0.010      # relative: #8's as-sintered method is not stated (geometric bias -1 +/- 1.5 % in the anneal layer)
RHO_TD = 7.62
ROUTE_850 = {"T_peak_C": 850.0, "heating_rate_C_per_min": 10.0, "dwell_min": 5.0, "P_MPa": 60.0,
             "pressure_mode": "cooling", "cooling_rate_C_per_min": 100.0, "load_ramp_min": 0.3, "degas": 1.0,
             "degas_completeness": 1.0}
ROUTE_BASIS = ("RPI panel recipe (signed PSF / timeline: 850 C, 10 C/min, 300 s, nominal 60 MPa, load on cooling, "
               "degassed 600 C); #7/#8 SPS logs were not released, so the delivered schedule is the panel plan, as for #6")
OBS = [
    {"id": "rho-C-7-archimedes", "body": "#7", "density_g_cm3": 7.01, "method": "Archimedes",
     "source": "RPI correction of the 7.10 typo (maintained CURRENT_STATE.md; operator check)", "sd_extra": 0.0},
    {"id": "rho-C-8-as-sintered", "body": "#8", "density_g_cm3": 7.07, "method": "not stated",
     "source": "Ziyi Wu (RPI) Gmail 1a15becba0000b94, 2026-10-20 (P5)", "sd_extra": SD_METHOD_8},
]


def load_tool(pkg):
    pkg = Path(pkg)
    loader = SourceFileLoader("ndfeb_physics_tool_c6_update", str(pkg / "ndfeb_physics_tool.py"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    tool = importlib.util.module_from_spec(spec)
    loader.exec_module(tool)
    return tool, tool.load(manifest=pkg / "TOOL_MANIFEST.json", physics=pkg / "physics.py", forward=pkg / "forward.py",
                           posterior=pkg / "posterior_tool.npz", microstructure=pkg / "microstructure.py",
                           coercivity_link=pkg / "coercivity_link.py")


def t5(y, mu, s, nu=5.0):
    z = (y - mu) / s
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - np.log(s)
            - (nu + 1) / 2 * np.log1p(z ** 2 / nu))


def lme(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def loglik(pm, obs, shared, rng, z_out=None):
    """Per density mechanism: log p(readings | stored draw), new preparation effects integrated. With z_out (a dict)
    and a shared preparation, also returns per draw the K latent effects and their log-likelihoods (z_out[m] =
    (zs, acc)), from which a draw's conditional posterior of the PREP-C850 effect is resampled."""
    r = pm._route(dict(ROUTE_850, prep="PREP-C850"))
    A = pm._arrays(r)
    out = {}
    for m in pm._mechs("density"):
        p = {k: v[:, None] for k, v in pm.post[m].items()}
        Kd = len(pm.post[m]["lnk"])
        zs = rng.standard_normal((Kd, K_LATENT))
        z2 = zs if shared else rng.standard_normal((Kd, K_LATENT))
        acc = np.zeros((Kd, K_LATENT))
        for j, o in enumerate(obs):
            y = o["density_g_cm3"] / RHO_TD
            z = zs if j == 0 else z2
            for k in range(K_LATENT):
                mu = pm._dens_mu(p, m, A, p["tau_prep"] * z[:, k:k + 1], r)[:, 0]
                s = np.sqrt(p["sig_rho"][:, 0] ** 2 + o["sd_extra"] ** 2)
                acc[:, k] += t5(y, mu, s)
        out[m] = lme(acc, 1)
        if z_out is not None:
            z_out[m] = (zs, acc)
    return out


def reweight(pm, ll):
    dw, lm, ess, ess_d = {}, {}, {}, {}
    for m, l in ll.items():
        base = np.log(pm.dw[m] / pm.dw[m].sum())
        tot = base + l
        mx = tot.max()
        lm[m] = float(mx + np.log(np.exp(tot - mx).sum()))
        w = np.exp(tot - lm[m])
        dw[m] = w
        ess[m] = float(1.0 / np.sum(w ** 2))
        inv, nd = pm._distinct(m)
        g = np.bincount(inv, weights=w, minlength=nd)
        ess_d[m] = float(g.sum() ** 2 / np.sum(g ** 2))
    z = np.log(sum(pm.w["density"][m] * math.exp(lm[m]) for m in ll))
    W = {m: float(pm.w["density"][m] * math.exp(lm[m] - z)) for m in ll}
    return dw, W, ess, ess_d


def predictive(pm, dw, W, n=8000, seed=0):
    """Density predictive (g/cm3) at the 850 C panel route for a NEW body of a NEW preparation under weights (dw, W)."""
    clone = object.__new__(type(pm))
    clone.__dict__.update(pm.__dict__)
    clone.dw = dict(pm.dw, **dw)
    clone.w = dict(pm.w, density=W)
    res = clone.predict(dict(ROUTE_850, prep="new"), n_draws=n, seed=seed)
    d = res["outputs"]["density_g_cm3"]
    return {k: float(d[k]) for k in ("q05", "q25", "median", "q75", "q95")}


def resample_with_batch(pm, dw, zinfo, rng):
    """Systematic resampling of each density mechanism's stored draws by dw; every resampled slot also draws its
    PREP-C850 effect from that draw's conditional over the K latent samples (so copies of one draw get their own z)."""
    idx, zpost = {}, {}
    for m, w in dw.items():
        n = len(w)
        u = (rng.random() + np.arange(n)) / n
        src = np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), n - 1)
        zs, acc = zinfo[m]
        a = acc[src] - acc[src].max(1, keepdims=True)
        pr = np.exp(a)
        pr /= pr.sum(1, keepdims=True)
        k = (rng.random(n)[:, None] > np.cumsum(pr, 1)).sum(1)
        idx[m] = src
        zpost[m] = zs[src, np.minimum(k, zs.shape[1] - 1)]
    return idx, zpost


def same_batch_predictive(pm, idx, zpost, W, n=8000, seed=1):
    """Density (g/cm3) of another body from the same 850 C batch (PREP-C850) at the panel route: per resampled draw
    mu(z) plus the draw's t5 scatter, mechanisms by W."""
    rng = np.random.default_rng(seed)
    r = pm._route(dict(ROUTE_850, prep="PREP-C850"))
    A = pm._arrays(r)
    mechs = list(W)
    counts = rng.multinomial(n, np.array([W[m] for m in mechs]) / sum(W.values()))
    out = []
    for m, c in zip(mechs, counts):
        j = rng.integers(0, len(idx[m]), c)
        src = idx[m][j]
        p = {k: v[src][:, None] for k, v in pm.post[m].items()}
        mu = pm._dens_mu(p, m, A, p["tau_prep"] * zpost[m][j][:, None], r)[:, 0]
        out.append(np.clip(mu + p["sig_rho"][:, 0] * rng.standard_t(5.0, c), 0.3, 1.02))
    x = np.concatenate(out) * RHO_TD
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(zip(("q05", "q25", "median", "q75", "q95"), map(float, q)))


def build_posterior(pkg, idx, zpost, W, out, provenance):
    """C7 posterior file: the C6 v2 arrays with each density mechanism's post__ arrays resampled (idx) and the fitted
    batch effect post__D*__z_prep_5 (PREP-C850) added; magnetic, C5/C6 layer and prior arrays unchanged."""
    z = np.load(Path(pkg) / "posterior_tool.npz", allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    arrays = {}
    for key in z.files:
        if key == "__meta__":
            continue
        grp, mech, par = key.split("__", 2)
        a = z[key]
        if grp == "post" and mech in idx:
            a = a[idx[mech]]
        arrays[key] = a
    n_prep = len(meta["density_preps"])
    for m, zp in zpost.items():
        arrays[f"post__{m}__z_prep_{n_prep}"] = zp.astype(z[f"post__{m}__tau_prep"].dtype)
    base = {k: meta.get(k) for k in ("version", "slice", "data_cutoff_step", "release_step")}
    meta["density_preps"] = meta["density_preps"] + ["PREP-C850"]
    meta["density_observations"] = meta["density_observations"] + [o["id"] for o in OBS]
    meta["weights"] = dict(meta["weights"], density=W)
    meta.update(version="ndfeb-sps-physics-posterior-20261023.8.0-C7", slice="C7", data_cutoff_step="P6",
                release_step="C7-20261023")
    meta["c7"] = dict(provenance, base=base, density_weights_before=provenance["density_weights_before"])
    np.savez(out / "posterior_tool.npz", __meta__=np.array(json.dumps(meta)), **arrays)


def main(pkg, out_dir):
    out = Path(out_dir)
    (out / "build").mkdir(parents=True, exist_ok=True)
    tool, pm = load_tool(pkg)
    rng = np.random.default_rng(SEED)
    runs = {"shared_preparation": (OBS, True), "separate_preparations": (OBS, False), "only_#7": (OBS[:1], True),
            "only_#8": (OBS[1:], True)}
    report = {"observations": OBS, "route": ROUTE_850, "route_basis": ROUTE_BASIS,
              "base_package_tool_version": tool.TOOL_VERSION, "base_posterior_sha256": pm.hashes["posterior"],
              "base_density_weights": pm.w["density"],
              "predictive_850_new_batch_before_g_cm3": predictive(pm, pm.dw, pm.w["density"])}
    keep = {}
    for i, (name, (obs, shared)) in enumerate(runs.items()):
        zinfo = {}
        ll = loglik(pm, obs, shared, np.random.default_rng([SEED, i]), z_out=zinfo)
        dw, W, ess, ess_d = reweight(pm, ll)
        report[name] = {"density_weights": W, "ess_stored": ess, "ess_distinct": ess_d,
                        "predictive_850_new_batch_after_g_cm3": predictive(pm, dw, W)}
        if shared:
            idx, zpost = resample_with_batch(pm, dw, zinfo, np.random.default_rng([SEED, i, 1]))
            report[name]["predictive_850_same_batch_g_cm3"] = same_batch_predictive(pm, idx, zpost, W)
            keep[name] = (idx, zpost, W)
    report["reproduction_check_c51prelim_v2"] = {
        "archived": {"median": 7.0376, "q05": 6.8447, "q95": 7.2777,
                     "what": "C5.1-prelim v2 (C4 data + #7 = 7.01, refit) forecast of #8 from the same 850 C batch, "
                             "frozen 2026-10-17T00:06:18Z; its package was not preserved"},
        "this_update_only_#7": report["only_#7"]["predictive_850_same_batch_g_cm3"],
        "note": "importance reweighting of the C6 v2 (= C5.0) density draws instead of a NUTS refit; agreement is a "
                "consistency check of the update method, not a new prediction"}
    idx, zpost, W = keep["shared_preparation"]
    build_posterior(pkg, idx, zpost, W, out, {
        "density_update": "importance reweighting of the C5.0 density draws on #7 (7.01, Archimedes) and #8 as "
                          "sintered (7.07, method not stated, +1 % relative sd), one shared new 850 C batch "
                          "(PREP-C850: the timeline's batch 3 is #7-#9); systematic resampling; PREP-C850 effect "
                          "drawn per slot from the draw's conditional",
        "density_weights_before": pm.w["density"], "ess_distinct": report["shared_preparation"]["ess_distinct"],
        "observations": OBS, "seed": SEED, "k_latent": K_LATENT, "base_posterior_sha256": pm.hashes["posterior"],
        "not_used": "#8 after 900 C (the anneal layer's observation); #1-#6 already in the C4/C5 fit"})
    (out / "build" / "DENSITY_UPDATE_REPORT.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1], sys.argv[2])
    for k in ("shared_preparation", "separate_preparations", "only_#7", "only_#8"):
        print(k, json.dumps(rep[k]))
    print(json.dumps(rep["reproduction_check_c51prelim_v2"], indent=1))
    print("before", rep["predictive_850_new_batch_before_g_cm3"])
