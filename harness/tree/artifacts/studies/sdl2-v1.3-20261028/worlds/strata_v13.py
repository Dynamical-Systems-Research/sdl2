#!/usr/bin/env python3
"""World properties at the release bar for v1.3 panel design (WORLD_DESIGN.md §3, §7). Owner side; computed from world
truth and the shipped supplied model only, never from any arm's outcome.

For one world: truth at 0.95 (truth_v13.world_truth), then
  feasible, best route, best P(all 3), passing count
  class         feasible: model_top_passes | canonical_passes | the lever group separating the nearest passing route
                from the live canonical route (worlds.GROUPS) | multi;
                infeasible: reliability_near_miss (a route reaches P(all 3) >= 0.5 but none 0.95) | near_miss_* (best
                median body >= 85 % of both limits) | clear_*, with _model_best / _best_elsewhere as worlds.strata
  binding       hci | bhmax | both (within 3 %) on the best route's median body
  path_decisive feasible only: the best route fails at 0.95 if the hidden magnetic path were the other one
  model_rank_first_passing   rank of the first passing route in the shipped supplied model's ranking (feasible only)
  ../sdl2-v1.2-20261026/python/bin/python -I worlds/strata_v13.py POOL_JSONL OUT_JSONL [--procs P]
     POOL_JSONL rows need "seed" and either "world" (a v1.3 or v1.2 config) or an eval-split seed to draw from
"""
import argparse
import copy
import json
import multiprocessing as mp
import sys
from pathlib import Path

V13 = Path(__file__).resolve().parents[1]
V12 = V13.parent / "sdl2-v1.2-20261026"
sys.path.insert(0, str(V13 / "src"))
sys.path.insert(0, str(V13 / "grader"))
sys.path.insert(0, str(V12 / "worlds"))
import sdl2_v13 as v13  # noqa: E402
import truth_v13 as T13  # noqa: E402
import worlds as WD  # noqa: E402
from ndfeb_sps.models import sdl2_v12 as v12  # noqa: E402

MEAS = {"density_body_sd_g_cm3": 0.02, "link_model_draws": 4000}
_RANK = None


def ranking():
    """The shipped supplied model's ranking (worlds/ranking/supplied_ranking_v13_*.jsonl), best first, ties by
    declared order, paths at equal odds (as worlds.supplied_top)."""
    global _RANK
    if _RANK is None:
        order = {v12.route_id(r): i for i, r in enumerate(v12.declared_routes())}
        rows = {}
        for f in sorted((V13 / "worlds/ranking").glob("supplied_ranking_v13_*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                rows[r["id"]] = 0.5 * (r["link"]["p_all"] + r["c5"]["p_all"])
        if len(rows) != len(order):
            raise RuntimeError(f"shipped ranking incomplete: {len(rows)} of {len(order)}")
        ids = sorted(rows, key=lambda k: (-rows[k], order[k]))
        _RANK = {k: i + 1 for i, k in enumerate(ids)}
    return _RANK


def ratio(med):
    return min(med["hci_kOe"] / v12.REQUIREMENT["hci_kOe"], med["bhmax_MGOe"] / v12.REQUIREMENT["bhmax_MGOe"])


def properties(seed, cfg):
    t = T13.world_truth(cfg)
    rows = t["routes"]
    rank = ranking()
    top = min(rank, key=rank.get)
    byid = {r["id"]: r for r in rows}
    best = byid[t["best"]]
    hr = best["median"]["hci_kOe"] / v12.REQUIREMENT["hci_kOe"]
    br = best["median"]["bhmax_MGOe"] / v12.REQUIREMENT["bhmax_MGOe"]
    out = {"seed": seed, "feasible": t["feasible"], "lps_only": t["lps_only"], "path": t["magnetic_path"],
           "best": t["best"], "best_p_all": t["best_p_all"], "n_passing": len(t["passing"]),
           "binding": "hci" if hr < 0.97 * br else ("bhmax" if br < 0.97 * hr else "both"),
           "best_median_ratio": max(ratio(r["median"]) for r in rows)}
    if t["feasible"]:
        passing = set(t["passing"])
        if top in passing:
            cls = "model_top_passes"
        elif WD.CANONICAL in passing:
            cls = "canonical_passes"
        else:
            cr = byid[WD.CANONICAL]["route"]
            near = min(passing, key=lambda rid: (sum(byid[rid]["route"][k] != cr[k] for k in WD.GROUPS),
                                                 [r["id"] for r in rows].index(rid)))
            changed = [k for k in WD.GROUPS if byid[near]["route"][k] != cr[k]]
            groups = {WD.GROUPS[k] for k in changed}
            cls = groups.pop() if len(groups) == 1 else "multi"
            out["nearest_passing_changed"] = changed
        out["class"] = cls
        out["model_rank_first_passing"] = min(rank[rid] for rid in passing)
        other = copy.deepcopy(cfg)
        c12 = other["v12"] if "v12" in other else other
        c12["magnetic_path"] = "c5" if c12["magnetic_path"] == "link" else "link"
        out["path_decisive"] = not T13.route_outcome(other, byid[t["best"]]["route"])["meets"]
    else:
        ratios = {r["id"]: ratio(r["median"]) for r in rows}
        best_set, _, _ = v12.best_route_set(ratios)
        suffix = "_model_best" if top in best_set else "_best_elsewhere"
        if max(r["p_all"] for r in rows) >= 0.5:
            out["class"] = "reliability_near_miss" + suffix
        elif out["best_median_ratio"] >= WD.NEAR_MISS:
            out["class"] = "near_miss" + suffix
        else:
            out["class"] = "clear" + suffix
    return out


def job(row):
    if "world" in row:
        cfg = row["world"]
    else:
        cfg = v13.draw("eval", f"strata-{row['seed']}", MEAS, world_seed=row["seed"])
    return properties(row["seed"], cfg)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("pool")
    a.add_argument("out")
    a.add_argument("--procs", type=int, default=1)
    x = a.parse_args()
    rows = [json.loads(line) for line in open(x.pool)]
    out = Path(x.out)
    done = {json.loads(line)["seed"] for line in out.open()} if out.exists() else set()
    todo = [r for r in rows if r["seed"] not in done]
    ranking()
    with mp.get_context("fork").Pool(x.procs) as pool, out.open("a") as fh:
        for r in pool.imap_unordered(job, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
