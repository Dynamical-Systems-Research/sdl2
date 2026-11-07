#!/usr/bin/env python3
"""SDL-2 route truth: world truth and a nominated route's outcome at the release bar.

Owner side. The physics, owner confirmation lots (common random numbers) and the 3 % median screen are the owner
world's (owner_world.py), called unchanged; only the reliability bar moves from the owner world's 0.5 to 0.95, with the
sequential Wilson estimator run around that bar and extended to 2,048 lots. At bar=0.5 with the owner world's lot schedule
(min 64, batch 16, max 256) these functions reproduce owner_world.world_truth and route_outcome exactly (checked by
`route_truth.py selftest`).

A route at the bar: P(all three bodies of one fresh preparation meet Hci >= 10 kOe and (BH)max >= 11 MGOe) >= bar.
A route whose zero-effect median body misses either limit by more than 3 % is not sampled: below the median, one body
meets with probability < 0.5, so three cannot meet with probability >= 0.125, far under either bar.

CLI (run with ../bridge/python/bin/python -I):
  route_truth.py selftest POOL.jsonl N     regression against owner_world at bar 0.5 on the first N rows' worlds, then the
                                         same worlds at 0.95
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from ndfeb_sps.models import campaign_world as campaign
from ndfeb_sps.models import owner_world as owner_world

STUDY = Path(__file__).resolve().parents[1]
BAR = 0.95
MIN_LOTS, LOT_BATCH, MAX_LOTS = 64, 64, 2048
Z = 1.96


def _owner_config(world_config):
    """Evaluation worlds carry the owner-world configuration under "v12"; owner-world configurations pass through."""
    if world_config.get("family") == owner_world.FAMILY:
        return world_config
    return world_config["v12"]


def wilson(k, n, z=Z):
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sequential_p_all(world, route, *, path, salt, bar=BAR, min_lots=MIN_LOTS, batch=LOT_BATCH, max_lots=MAX_LOTS):
    """owner_world.sequential_p_all with the stopping bar as a parameter: stop once the 95% Wilson interval excludes the bar,
    or at max_lots (then the point estimate decides, and the row is flagged)."""
    k, n = owner_world.route_p_all(world, route, path=path, salt=salt, lots=min_lots, start=0)
    while True:
        lo, hi = wilson(k, n)
        if lo > bar or hi < bar or n >= max_lots:
            return k, n, not (lo > bar or hi < bar)
        dk, dn = owner_world.route_p_all(world, route, path=path, salt=salt, lots=batch, start=n)
        k, n = k + dk, n + dn


def world_truth(world_config, *, bar=BAR, min_lots=MIN_LOTS, batch=LOT_BATCH, max_lots=MAX_LOTS):
    """owner_world.world_truth at `bar`: every declared route's median body; P(all 3) wherever the screen admits it."""
    cfg = _owner_config(world_config)
    base, path = cfg["base_world"], cfg["magnetic_path"]
    world = campaign.World(base)
    salt = owner_world.owner_salt(cfg)
    routes = owner_world.declared_routes()
    req = owner_world.REQUIREMENT
    rows, max_raw_hci = [], 0.0
    for route, med in zip(routes, owner_world.median_table(cfg, routes)):
        max_raw_hci = max(max_raw_hci, med["raw_hci_kOe"])
        row = {"route": route, "id": owner_world.route_id(route), "median": med, "p_all": 0.0, "lots": 0, "straddles": False,
               "lps_extrapolated": bool(route["anneal"] != "none" and world.amech == "LPS"
                                        and med["rho_made_rel"] < owner_world.LPS_START_FLOOR_REL)}
        if owner_world.meets(med, owner_world.SCREEN_MARGIN):
            k, n, straddles = sequential_p_all(world, route, path=path, salt=salt, bar=bar, min_lots=min_lots,
                                               batch=batch, max_lots=max_lots)
            row.update(p_all=k / n, lots=n, straddles=straddles)
        rows.append(row)
    passing = [r for r in rows if r["p_all"] >= bar]
    best = max(rows, key=lambda r: (r["p_all"], min(r["median"]["hci_kOe"] / req["hci_kOe"],
                                                    r["median"]["bhmax_MGOe"] / req["bhmax_MGOe"])))
    return {"bar": bar, "feasible": bool(passing), "passing": [r["id"] for r in passing],
            "best": best["id"], "best_p_all": best["p_all"], "routes": rows, "max_raw_hci_kOe": max_raw_hci,
            "magnetic_path": path, "draw": base["draw"],
            "straddling_routes": [r["id"] for r in rows if r["straddles"]],
            "lps_only": bool(passing) and all(r["lps_extrapolated"] for r in passing)}


def route_outcome(world_config, route, *, bar=BAR, min_lots=MIN_LOTS, batch=LOT_BATCH, max_lots=MAX_LOTS):
    """Rung-1 truth for one nominated route at `bar`, on the same owner lots and estimator as world_truth."""
    cfg = _owner_config(world_config)
    route = owner_world.canonical_route(route)
    base, path = cfg["base_world"], cfg["magnetic_path"]
    med = owner_world.median_table(cfg, [route])[0]
    if not owner_world.meets(med, owner_world.SCREEN_MARGIN):
        return {"route": owner_world.route_id(route), "bar": bar, "p_all": 0.0, "lots": 0, "meets": False, "screened": True,
                "straddles": False}
    k, n, straddles = sequential_p_all(campaign.World(base), route, path=path, salt=owner_world.owner_salt(cfg), bar=bar,
                                       min_lots=min_lots, batch=batch, max_lots=max_lots)
    return {"route": owner_world.route_id(route), "bar": bar, "p_all": k / n, "lots": n, "meets": k / n >= bar,
            "straddles": straddles}


def _strip(t):
    return {k: v for k, v in t.items() if k not in ("version", "routes", "straddling_routes", "bar")}


def selftest(pool, n):
    """Regression: at bar 0.5 with the owner world's lot schedule, identical to owner_world.world_truth (every route's p_all and lots,
    feasibility, best). Then the same worlds at the release bar."""
    rows = [json.loads(line) for line in open(pool)]
    feas = [r for r in rows if r["feasible"] and not r["lps_only"]]
    near = [r for r in rows if not r["feasible"] and r["best_ratio"] >= 0.85]
    worlds = [r["world"] for pair in zip(feas, near) for r in pair][:n]   # alternate feasible / near-miss
    out = []
    for i, w in enumerate(worlds):
        t0 = time.time()
        ref = owner_world.world_truth(w)
        mine = world_truth(w, bar=0.5, min_lots=owner_world.MIN_LOTS, batch=owner_world.LOT_BATCH, max_lots=owner_world.MAX_LOTS)
        same_routes = all(a["id"] == b["id"] and a["p_all"] == b["p_all"] and a["lots"] == b["lots"]
                          for a, b in zip(ref["routes"], mine["routes"]))
        same = same_routes and _strip(ref) == _strip(mine)
        t1 = time.time()
        hi = world_truth(w)
        t2 = time.time()
        out.append({"world": i, "identical_at_0_5": same, "feasible_0_5": ref["feasible"],
                    "best_p_all_0_5": ref["best_p_all"], "feasible_0_95": hi["feasible"],
                    "best_p_all_0_95": hi["best_p_all"], "passing_0_5": len(ref["passing"]),
                    "passing_0_95": len(hi["passing"]), "straddling_0_95": len(hi["straddling_routes"]),
                    "max_lots_0_95": max(r["lots"] for r in hi["routes"]), "s_0_5_pair": round(t1 - t0, 1),
                    "s_0_95": round(t2 - t1, 1)})
        print(json.dumps(out[-1]), flush=True)
    assert all(r["identical_at_0_5"] for r in out), "regression against owner_world.world_truth failed"
    print("SELFTEST OK: identical to owner_world at bar 0.5 on", len(out), "worlds")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        selftest(sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
