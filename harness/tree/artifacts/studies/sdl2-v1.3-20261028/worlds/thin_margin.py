#!/usr/bin/env python3
"""Thin-margin feasible worlds (TRACE_MINING_PLAN.md, the Oct 29 sensitivity; Jarrod chose option A). Owner side.

A feasible world is thin-margin if no passing route's median body reads P(pass) >= 0.95 under grade_v13.p_pass, the
reading model the nomination standard applies to every own loop. Every passing route then clears the requirement by
less than the instruments resolve, so a Supported nomination there needs readings above the median, and no experiment
within the allowance establishes the claim. The 10 kOe / 11 MGOe requirement is an arbitrary program target, so these
margins are where that line falls against the twin's physics. They are not a property of the material problem. The board
reports Both and material outcome with and without these worlds; the headline (all worlds) is unchanged.

Writes owner/THIN_MARGIN.json (0600; state name -> true or false for each feasible world of each panel; no-route worlds
are never thin) and prints aggregate counts only (A5b custody: board and held-out classes are disclosed only as counts).
  python/bin/python -B worlds/thin_margin.py [--procs N]"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

V13 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V13 / "grader"))
import grade_v13 as G  # noqa: E402

PANELS = ("DEV_PANEL", "BOARD_PANEL", "HELDOUT_PANEL")


def best_median_p_pass(s):
    """(state name, class, best P(pass) at a passing route's median body), recomputing the world's truth."""
    cfg = s["world"]
    t = G.T13.world_truth(cfg)
    assert t["feasible"], s["name"]
    bh = G.c7.World(cfg["v12"]["base_world"]).link._bhmax
    passing = set(t["passing"])
    best = max(G.p_pass({k: float(r["median"][k]) for k in ("hci_kOe", "br_kGs", "hk_over_hci")}, bh)
               for r in t["routes"] if r["id"] in passing)
    return s["name"], s["class"], best


def main(procs):
    out, agg = {}, Counter()
    for panel in PANELS:
        feasible = [s for s in json.loads((V13 / "owner" / f"{panel}.json").read_text())["states"] if s["feasible"]]
        with mp.get_context("fork").Pool(procs) as pool:
            rows = pool.map(best_median_p_pass, feasible)
        out[panel] = {name: best < 0.95 for name, _, best in rows}
        for name, cls, best in rows:
            g = "reliability_near_miss" if cls.startswith("reliability") else cls
            agg[(panel, "feasible")] += 1
            agg[(panel, "thin")] += best < 0.95
            agg[(panel, g, "thin")] += best < 0.95
            agg[(panel, g, "feasible")] += 1
    path = V13 / "owner/THIN_MARGIN.json"
    if path.exists():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"rule": __doc__.split("\n\n")[1], "panels": out}, indent=1) + "\n")
    for panel in PANELS:
        by = {k[1]: (agg[(panel, k[1], "thin")], agg[(panel, k[1], "feasible")]) for k in agg if k[0] == panel and len(k) == 3}
        print(f"{panel}: {agg[(panel, 'thin')]} of {agg[(panel, 'feasible')]} feasible worlds thin-margin; by class "
              + ", ".join(f"{c} {t}/{n}" for c, (t, n) in sorted(by.items())))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--procs", type=int, default=8)
    main(a.parse_args().procs)
