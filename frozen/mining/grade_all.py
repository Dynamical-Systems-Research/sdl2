"""Grade every episode under one or more roots with the frozen grader, computing each world's truth once and sharing it
across the arms that ran that world. Owner only. Writes GRADES.jsonl (one row per episode) and WORLDS.jsonl (one row
per world: truth summary and the pool row's decisive-lever record).
  ../python/bin/python -B grade_all.py OUT_DIR ROOT [ROOT ...] [--procs N]"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "grader"))
sys.path.insert(0, str(HERE))
import grade_episode as G  # noqa: E402
import traces  # noqa: E402

POOLS = {}   # panels are JSON (owner/*_PANEL.json); the class comes from PREPARATION.json


def pool_rows(panel, seeds):
    """The pool rows for these seeds (streamed: the sealed pool is 155 MB)."""
    want, out = set(seeds), {}
    f = STUDY / "owner" / POOLS[panel]
    if f.exists() and want:
        for line in f.open():
            r = json.loads(line)
            if r.get("seed") in want:
                out[r["seed"]] = r
                if len(out) == len(want):
                    break
    return out


def grade_world(job):
    """All episodes of one world: truth once, then each episode's grade."""
    world_key, eps, pool_row = job
    cfg = json.loads((Path(eps[0]) / "simulation-magnetic/reference_world.json").read_text())
    truth = G.route_truth.world_truth(cfg)   # the 0.95 bar
    best, best_ratio = G.best_routes(truth)
    world = {"world": world_key, "feasible": truth["feasible"], "n_passing_routes": len(truth["passing"]),
             "passing_routes": sorted(truth["passing"]), "best_routes": sorted(best), "best_median_ratio": best_ratio,
             "magnetic_path": cfg["v12"]["magnetic_path"],
             "nearest_passing": (pool_row or {}).get("nearest_passing"), "class": None}
    rows = []
    for ep in eps:
        ident = traces.identity(ep)
        world["class"] = world["class"] or json.loads((Path(ep) / "PREPARATION.json").read_text())["origin"].get("class")
        rec = traces.receipt(ep) or {}
        try:
            c, obs, ex, entries = G.records(Path(ep))
            claim, err = G.read_decision(Path(ep))
            g = G.evaluate(c, obs, ex, entries, claim, err, truth=truth, diagnostic_posterior=False)
            g["forecasts"] = {"counts": g["forecasts"]["counts"]}   # observed, not graded: counts only, no skill scores
            g["claim"] = claim
            if claim and claim["form"] in ("nominate", "restrict"):   # credits to a supported decision
                g["credits_to_supported"] = G.credits_to_supported(c, obs, ex, entries, claim, truth)
        except Exception as x:  # noqa: BLE001 - an ungradable record is reported, never dropped
            g = {"grade_error": repr(x)[:400]}
        rows.append({**ident, "path": str(Path(ep).resolve()), "world_key": world_key, "receipt_status": rec.get("status"),
                     "termination": rec.get("termination_cause"), **g})
    return world, rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("roots", nargs="+")
    a.add_argument("--procs", type=int, default=16)
    x = a.parse_args()
    eps = sorted(p for r in x.roots for p in Path(r).iterdir()
                 if (p / "simulation-magnetic/reference_world.json").exists() and (p / "PREPARATION.json").exists())
    by_world = {}
    for ep in eps:
        key = hashlib.sha256((ep / "simulation-magnetic/reference_world.json").read_bytes()).hexdigest()[:16]
        by_world.setdefault(key, []).append(str(ep))
    seeds = {}
    for key, group in by_world.items():
        o = traces.identity(group[0])
        seeds.setdefault(o["panel"], {})[o["seed"]] = key
    rows_by_key = {}
    for panel, m in seeds.items():
        if panel in POOLS:
            for seed, row in pool_rows(panel, m).items():
                rows_by_key[m[seed]] = row
    out = Path(x.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, g, rows_by_key.get(k)) for k, g in by_world.items()]
    with ProcessPoolExecutor(x.procs) as pool, open(out / "GRADES.jsonl", "w") as fg, open(out / "WORLDS.jsonl", "w") as fw:
        for world, rows in pool.map(grade_world, jobs):
            fw.write(json.dumps(world) + "\n")
            for r in rows:
                fg.write(json.dumps(r, default=float) + "\n")
    print(f"graded {len(eps)} episodes on {len(by_world)} worlds -> {out}")


if __name__ == "__main__":
    main()
