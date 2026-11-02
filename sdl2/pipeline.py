"""Rebuild the leaderboard from the released records with the frozen code.

The run directory mirrors the study's layout: a copy of `frozen/`, the leaderboard manifest under `leaderboard/` and
hidden-truth files under `owner/`. The frozen scripts run unchanged and every output is compared with the published file.
"""
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import FROZEN, leaderboard

QUEUE = "leaderboard/QUEUE_PHASE_D.json"
PAIRS = (("openai/gpt-6-astra", "bo"), ("openai/gpt-6-astra", "doe"), ("deepseek/deepseek-v4.1-flash-1017", "bo"))
LEADERBOARD_FILES = ("GRADES.jsonl", "WORLDS.jsonl", "FEATURES.jsonl", "SUPPORTED.jsonl", "C2S.jsonl", "FLOOR.json",
                     "FLOOR_SUPPORTED.jsonl", "RESULTS.json", "HEADLINE.md", "REPORT.md", "SENSITIVITY.json",
                     "SENSITIVITY.md", "PAIR_gpt-6-astra__bo.json", "PAIR_gpt-6-astra__doe.json",
                     "PAIR_deepseek-v4.1-flash-1017__bo.json")


def python():
    return sys.executable


def world_key(config):
    return hashlib.sha256(Path(config).read_bytes()).hexdigest()[:16]


def study(data, out):
    """Create the run directory: the frozen code and the leaderboard manifest."""
    run = Path(out) / "study"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(FROZEN, run, ignore=shutil.ignore_patterns("__pycache__"))
    (run / "leaderboard").mkdir()
    shutil.copyfile(Path(data) / "board/QUEUE_PHASE_D.json", run / QUEUE)
    (run / "owner").mkdir()
    return run


def episodes(data):
    """Episode directories in manifest order."""
    q = json.loads((Path(data) / "board/QUEUE_PHASE_D.json").read_text())["entries"]
    eps = [Path(data) / "traces" / e["name"] for e in q]
    missing = [e.name for e in eps if not (e / "simulation-magnetic/reference_world.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest episodes are not unpacked under {data}/traces, e.g. {missing[:3]}")
    return eps


def root(data, run):
    """Link every episode under mining/root_board_all, as the study did."""
    r = run / "mining/root_board_all"
    r.mkdir()
    for ep in episodes(data):
        os.symlink(ep.resolve(), r / ep.name)
    return r


def _world_truth(job):
    package, frozen, cfg_path = job
    leaderboard.install(package, frozen)
    import grade_v13 as G
    cfg = json.loads(Path(cfg_path).read_text())
    t = G.T13.world_truth(cfg)
    bh = G.c7.World(cfg["v12"]["base_world"]).link._bhmax
    passing = set(t["passing"])
    best = max((G.p_pass({k: float(r["median"][k]) for k in ("hci_kOe", "br_kGs", "hk_over_hci")}, bh)
                for r in t["routes"] if r["id"] in passing), default=None)
    return cfg_path, t, best


def truth(data, out, package, procs, frozen):
    """Recompute the truth of all 8,064 routes in every world and compare it with the released table. Also returns the
    best P(pass) at a passing route's median magnet, the input to the thin-margin rule."""
    worlds = {}
    for ep in episodes(data):
        cfg = ep / "simulation-magnetic/reference_world.json"
        worlds.setdefault(world_key(cfg), str(cfg))
    tdir = Path(out) / "truth"
    tdir.mkdir(parents=True, exist_ok=True)
    rows, keys = {}, {v: k for k, v in worlds.items()}
    with ProcessPoolExecutor(procs, initializer=leaderboard.install, initargs=(str(package), str(frozen))) as pool:
        for cfg_path, t, best in pool.map(_world_truth, [(str(package), str(frozen), p) for p in worlds.values()]):
            key = keys[cfg_path]
            (tdir / f"{key}.json").write_text(json.dumps(t, sort_keys=True) + "\n")
            released = json.loads((Path(data) / "worlds/truth" / f"{key}.json").read_text())
            rows[key] = {"identical_to_released": released == t, "feasible": t["feasible"],
                         "best_median_p_pass": best}
    return rows


def thin_margin(data, run, truth_rows):
    """Write owner/THIN_MARGIN.json: a feasible world is thin-margin when no passing route's median magnet reads
    P(pass) >= 0.95 (worlds/thin_margin.py)."""
    state = {}
    for ep in episodes(data):
        state[world_key(ep / "simulation-magnetic/reference_world.json")] = \
            json.loads((ep / "PREPARATION.json").read_text())["origin"]["state"]
    flags = {state[k]: r["best_median_p_pass"] < 0.95 for k, r in truth_rows.items() if r["feasible"]}
    (run / "owner/THIN_MARGIN.json").write_text(json.dumps({"panels": {"BOARD_PANEL": flags}}, indent=1) + "\n")
    return flags


def floors(data, run, mining_out):
    """Write the two zero-experiment floors: restrict in every world, or nominate the supplied model's top route
    untested."""
    import truth_v13 as T13
    from ndfeb_sps.models import sdl2_v12 as v12
    declared = v12.declared_routes()
    order = {v12.route_id(r): i for i, r in enumerate(declared)}
    rank = {}
    for f in sorted((Path(data) / "worlds/ranking").glob("supplied_ranking_v13_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rank[r["id"]] = 0.5 * (r["link"]["p_all"] + r["c5"]["p_all"])
    assert len(rank) == len(order), "supplied ranking incomplete"
    top_id = sorted(rank, key=lambda k: (-rank[k], order[k]))[0]
    top = next(r for r in declared if v12.route_id(r) == top_id)
    top = json.loads(json.dumps(v12.canonical_route(top)))
    entries = json.loads((run / QUEUE).read_text())["entries"]
    q_order = {e["state"]: e["order"] for e in entries}
    worlds = {r["world"]: r for r in map(json.loads, (mining_out / "WORLDS.jsonl").open())}
    by_state = {}
    for ep in episodes(data):
        st = json.loads((ep / "PREPARATION.json").read_text())["origin"]["state"]
        by_state.setdefault(st, ep / "simulation-magnetic/reference_world.json")
    rows = []
    for st, cfg_path in by_state.items():
        cfg = json.loads(cfg_path.read_text())
        feasible = worlds[world_key(cfg_path)]["feasible"]
        try:
            top_ok = bool(T13.route_outcome(cfg, top)["meets"])
        except (KeyError, ValueError, TypeError):
            top_ok = False
        rows.append({"order": q_order[st], "state": st, "feasible": feasible, "restrict_all_outcome": not feasible,
                     "top_route_outcome": top_ok})
    # FLOOR.json is in queue order, FLOOR_SUPPORTED.jsonl in world order
    (mining_out / "FLOOR.json").write_text(json.dumps({"top_route": top, "rows": sorted(rows, key=lambda r: r["order"])},
                                                      indent=1) + "\n")
    (mining_out / "FLOOR_SUPPORTED.jsonl").write_text("".join(
        json.dumps({"episode": f"floor-{policy}-o{r['order']:02d}", "standard": f"K{k}", "justified": False}) + "\n"
        for r in sorted(rows, key=lambda r: r["state"]) for policy in ("restrict", "top") for k in (30, 100, 200)))


def env(run):
    """Environment for the frozen scripts: the run directory's twin package and v1.2 worlds module."""
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(run), str(run / "worlds_v12")])}


def sh(args, cwd, run):
    subprocess.run(args, cwd=cwd, check=True, env=env(run))


def rebuild_leaderboard(data, out, package, procs):
    data, out, package = Path(data).resolve(), Path(out).resolve(), Path(package).resolve()
    run = study(data, out)
    rows = truth(data, out, package, procs, run)
    thin_margin(data, run, rows)
    r = root(data, run)
    mining = run / "mining"
    out_board = mining / "out_board"
    # Grade every episode with the frozen grade_all.py; its workers load the released package
    leaderboard.install(package, frozen=run)
    import grade_all
    import supported
    grade_all.ProcessPoolExecutor = functools.partial(ProcessPoolExecutor, initializer=leaderboard.install,
                                                      initargs=(str(package), str(run)))
    argv, cwd = sys.argv, os.getcwd()
    try:
        os.chdir(mining)
        sys.argv = ["grade_all.py", "out_board", r.name, "--procs", str(procs)]
        grade_all.main()
    finally:
        sys.argv = argv
        os.chdir(cwd)
    # Leaderboard tables: features, Supported, floors, report, paired comparisons, sensitivity
    py = [python(), "-B"]
    sh(py + ["features.py", "out_board", "--procs", str(procs)], mining, run)
    supported.main(out_board)
    floors(data, run, out_board)
    with open(out_board / "report.log", "w") as log:
        subprocess.run(py + ["report.py", "out_board", "../" + QUEUE, "31", "--justified", "out_board/SUPPORTED.jsonl",
                             "--declared", "K100", "--credits-to-justified", "out_board/C2S.jsonl",
                             "--floor-justified", "out_board/FLOOR_SUPPORTED.jsonl"],
                       cwd=mining, check=True, stdout=log, stderr=subprocess.STDOUT, env=env(run))
    for a, b in PAIRS:
        sh(py + ["report.py", "out_board", "../" + QUEUE, "31", "--justified", "out_board/SUPPORTED.jsonl",
                 "--declared", "K100", "--pair", a, b], mining, run)
    sh(py + ["sensitivity.py", "out_board"], mining, run)
    return {"truth": rows, "out_board": str(out_board)}


def compare(out_board, published):
    """Compare each output with the published file: GRADES.jsonl without the local `path` field, the rest byte for
    byte."""
    res = {}
    for f in LEADERBOARD_FILES:
        a, b = Path(out_board) / f, Path(published) / f
        if not a.exists() or not b.exists():
            res[f] = "missing"
            continue
        if f == "GRADES.jsonl":
            load = lambda p: [{k: v for k, v in json.loads(l).items() if k != "path"} for l in p.open()]  # noqa: E731,E741
            res[f] = "identical without path" if load(a) == load(b) else "DIFFERS"
        else:
            res[f] = "identical" if a.read_bytes() == b.read_bytes() else "DIFFERS"
    return res
