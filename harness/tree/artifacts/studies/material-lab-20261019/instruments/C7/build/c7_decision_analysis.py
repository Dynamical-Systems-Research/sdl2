"""C7 decision analysis: which observation separates the alternatives, and how the answer changes the next action.
Owner side; numpy only; reads the C7 package. Run (from the package): python3 -B build/c7_decision_analysis.py
Writes build/DECISION_ANALYSIS.json. Every number here is a conditional model calculation; none is a measurement.

  1  retained lower-density pieces through the same (unconfirmed) program as #8: predictions per anneal mechanism and
     how much one same-piece, same-method before/after reading would tell apart (from build/ANNEAL_FIT_REPORT.json)
  2  a matched as-sintered / annealed magnetic comparison on #8's material (7.07 g/cm3 as sintered): paired per-draw
     ratios after / before for Br, Hci and (BH)max on each magnetic path; the paths' Br responses point opposite ways
  3  the joint target by route family on both paths, at room temperature and at declared operating temperatures
  4  what decides the coercivity leg on the link path (boundary-film magnetization, grain, coverage)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PKG = Path(__file__).resolve().parents[1]
loader = SourceFileLoader("ndfeb_physics_tool_c7_analysis", str(PKG / "ndfeb_physics_tool.py"))
spec = importlib.util.spec_from_loader(loader.name, loader)
tool = importlib.util.module_from_spec(spec)
loader.exec_module(tool)
pm = tool.load()
BASE = {"T_peak_C": 850.0, "pressure_mode": "cooling", "P_MPa": 60.0, "dwell_min": 5.0}


def q(x, levels=(0.05, 0.5, 0.95)):
    return [float(v) for v in np.quantile(np.asarray(x, float), levels)]


def paired_magnetic(n=6000, seed=23):
    """C5 path: the same draws with and without the anneal (paired by the tool's own generator; the anneal layer adds
    no C5 randomness). Link path: one flat link call per stage with one generator seed, so the link's own draws are
    paired too."""
    out = {}
    for stage_key, anneal in (("after_900", "900"), ("after_900_520", "900+520")):
        r0 = dict(BASE)
        r1 = dict(BASE, anneal=anneal, anneal_hold_h="as_#8")
        a = pm.predict(r0, n_draws=n, seed=seed, rho_observed=7.07, include_draws=True)["draws"]
        b = pm.predict(r1, n_draws=n, seed=seed, rho_observed=7.07, include_draws=True)["draws"]
        c5 = {k: np.asarray(b[k], float) / np.asarray(a[k], float) for k in ("Br_kGs", "Hci_kOe", "BHmax_MGOe")}
        rr, sims, _, mlab, _, ind = pm._simulate(r1, n, seed, "post", rho_observed=7.07, sem=True, c6=True)
        st = pm._link_state(rr, sims, mlab, ind)
        flats = {}
        for s in ("as_sintered", stage_key):
            nd = st["nd_rich"][s]
            flats[s] = {"grain_median_um": st["grain_model_um"][s], "grain_sigma_ln": st["grain_sigma_ln"],
                        "density_rel": st["rho_rel_by_stage"][s],
                        "nd_rich_area_fraction": nd["nd_rich_area_fraction"], "pocket_share": nd["pocket_share"],
                        "faint_line_coverage": nd["faint_line_coverage"],
                        "pocket_share_as_sintered": st["nd_rich"]["as_sintered"]["pocket_share"],
                        "faint_line_coverage_as_sintered": st["nd_rich"]["as_sintered"]["faint_line_coverage"],
                        "coverage_sem_lower": np.full(n, np.nan), "stage": np.full(n, s),
                        "alignment_cos_theta": np.full(n, 0.5), "ln_hci_chem_offset": st["ln_hci_chem_offset"]}
        lo = {s: pm.link._link_flat(f, np.random.default_rng([seed, 5])) for s, f in flats.items()}
        link = {}
        for k, key in (("Br_kGs", "br_kGs"), ("Hci_kOe", "hci_kOe"), ("BHmax_MGOe", "bhmax_MGOe")):
            link[k] = lo[stage_key][key]["marginal"] / lo["as_sintered"][key]["marginal"]
        rows = {}
        for k in ("Br_kGs", "Hci_kOe", "BHmax_MGOe"):
            rows[k] = {"c5_ratio_q05_q50_q95": q(c5[k]), "link_ratio_q05_q50_q95": q(link[k]),
                       "c5_share_ratio_below_1": float(np.mean(c5[k] < 1.0)),
                       "link_share_ratio_below_1": float(np.mean(link[k] < 1.0))}
        dens = np.asarray(b["density_rel"], float) / np.asarray(a["density_rel"], float)
        rows["density_ratio_q05_q50_q95"] = q(dens)
        # separability of the two paths' Br predictions by a matched pair read with relative sd s per reading
        sep = {}
        for s in (0.02, 0.05):
            noise = np.random.default_rng(seed).normal(0, s * np.sqrt(2), n)
            x5, xl = np.log(c5["Br_kGs"]) + noise, np.log(link["Br_kGs"]) + noise
            thr = 0.5 * (np.median(x5) + np.median(xl))
            sep[f"reading_rel_sd_{s:g}"] = {"threshold_ln_ratio": float(thr),
                                            "c5_world_classified_c5": float(np.mean(x5 < thr)),
                                            "link_world_classified_link": float(np.mean(xl >= thr))}
        rows["br_response_separability"] = sep
        out[stage_key] = rows
    return out


ROUTE_FAMILIES = {
    "R0 current isotropic SPS (850 C, 5 min, load on cooling, 4 um), as sintered": dict(BASE),
    "R0a the same after 900 + 520 C / 2 h each": dict(BASE, anneal="900+520"),
    "R1 aligned body (<cos> 0.95, declared), 4 um, after 900 + 520 C": dict(BASE, alignment_cos=0.95, anneal="900+520"),
    "R2 aligned, 2 um feedstock (declared), after 900 + 520 C": dict(BASE, alignment_cos=0.95, anneal="900+520",
                                                                     feedstock_d50_um=2.0),
    "R3 aligned, 1 um feedstock (declared), after 900 + 520 C": dict(BASE, alignment_cos=0.95, anneal="900+520",
                                                                     feedstock_d50_um=1.0),
    "R3c R3 with complete degassing (declared)": dict(BASE, alignment_cos=0.95, anneal="900+520",
                                                      feedstock_d50_um=1.0, degas_completeness="complete"),
}


def joint_tables(n_worlds=3000, bodies=12, seed=31):
    out = {}
    for label, route in ROUTE_FAMILIES.items():
        row = {}
        for T in (20.0, 100.0, 150.0):
            spec = {"Hci_kOe_min": 20.0, "BHmax_MGOe_min": 45.0}
            if T != 20.0:
                spec["T_operating_C"] = T
            j = pm.joint_target(route, spec=spec, n_worlds=n_worlds, bodies_per_world=bodies, seed=seed)
            row[f"{T:g}C"] = {p: {"p_one_body": v["p_one_body"], "p_all_3_one_prep": v["p_all_3_bodies_x_1_preparation"],
                                  "p_Hci": v["marginals"].get("Hci_kOe_min"), "p_BHmax": v["marginals"].get("BHmax_MGOe_min"),
                                  "median_Hci_kOe": v["medians"]["Hci_kOe"], "median_BHmax_MGOe": v["medians"]["BHmax_MGOe"],
                                  **({"by_jgb_T": v["by_jgb_T"]} if "by_jgb_T" in v else {})}
                              for p, v in j["paths"].items()}
            if T == 20.0:
                row["out_of_hull_magnetic"] = j["out_of_hull"]["magnetic"]["outside_features"]
                row["structural"] = {k: j["structural"][k] for k in ("Br_needed_kGs", "Br_bound_kGs_Js_rho_cos")}
        out[label] = row
    return out


def coercivity_leg(n_worlds=3000, seed=37):
    """P(Hci >= 20 kOe) alone on the aligned 1 um route by path, with the link split by boundary-film magnetization."""
    j = pm.joint_target(ROUTE_FAMILIES["R3 aligned, 1 um feedstock (declared), after 900 + 520 C"],
                        spec={"Hci_kOe_min": 20.0}, n_worlds=n_worlds, bodies_per_world=12, seed=seed)
    return {p: {"p_Hci_ge_20": v["p_one_body"], "median_Hci_kOe": v["medians"]["Hci_kOe"],
                **({"by_jgb_T": v["by_jgb_T"], "link_flags_share": v["link_flags_share"]} if "by_jgb_T" in v else {}),
                "by_magnetic_mechanism": v["by_magnetic_mechanism"]} for p, v in j["paths"].items()}


def main():
    fit = json.loads((PKG / "build/ANNEAL_FIT_REPORT.json").read_text())
    res = {"tool_version": tool.TOOL_VERSION, "artifact_sha256": pm.hashes,
           "status": "conditional model calculations; no current-panel magnetic measurement exists",
           "1_retained_pieces_same_program_as_8": {
               "mechanism_weights": fit["mechanism_weights"],
               "predictions_true_density_g_cm3": {b: {m: v[m]["true_g_cm3"] for m in ("LPS", "CLOSED", "CAPACITY", "NULL")}
                                                  for b, v in fit["retained_piece_predictions_same_program_as_8"].items()},
               "information_about_mechanism": fit["information_from_retained_piece_anneal"],
               "nuisance_prior_sensitivity": fit["nuisance_prior_sensitivity"]},
           "2_matched_magnetic_comparison_8": paired_magnetic(),
           "3_joint_target_by_route_family": joint_tables(),
           "4_coercivity_leg_aligned_1um": coercivity_leg()}
    (PKG / "build/DECISION_ANALYSIS.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["2_matched_magnetic_comparison_8"], indent=1)[:3000])
    for k, v in r["3_joint_target_by_route_family"].items():
        print(k, {T: {p: (round(x["p_one_body"], 4), round(x["p_all_3_one_prep"], 4), round(x["p_Hci"], 4),
                          round(x["p_BHmax"], 4), round(x["median_Hci_kOe"], 1), round(x["median_BHmax_MGOe"], 1))
                      for p, x in v[T].items()} for T in ("20C", "100C")})
    print(json.dumps(r["4_coercivity_leg_aligned_1um"], indent=1)[:1500])
