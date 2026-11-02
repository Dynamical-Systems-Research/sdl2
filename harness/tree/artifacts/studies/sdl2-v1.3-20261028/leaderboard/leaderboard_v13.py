"""SDL-2 v1.3 Phase C (built from ../sdl2-v1.2-20261026/leaderboard/leaderboard.py by leaderboard/port_leaderboard.py;
diff in leaderboard/leaderboard_v13.diff). SDL2_PANEL=board (default): every D14 arm once on each of the 32 board
states (C1); SDL2_PANEL=heldout: the DeepSeek V4.1 Flash 1017 base on the 96 held-out states (C2, D4). Episodes run
under /lab (D12). The v1.2 text follows for the record.
The sealed leaderboard: every arm once on every sealed state (artifacts/proposals/sdl2-release-20261026/
SEALED_PANEL_AND_LEADERBOARD.md §3). Owner only; README.md is the operator procedure.
  ../python/bin/python -B leaderboard.py build              writes QUEUE.json once: 96 states x the arms below
  ../python/bin/python -B leaderboard.py run N [K/S]        runs every entry not yet started, N at a time; on two machines
                                                            each runs one shard of whole states (0/2 and 1/2)
  ../python/bin/python -B leaderboard.py status [K/S]       progress and cost per arm (no classes, worlds or outcomes)
  ../python/bin/python -B leaderboard.py resume NAME        a stopped model episode, from where it stopped

The queue runs the policies first, then the longest episodes first (ORDER), in a shuffled state order within each
arm. Each episode is prepared just before it runs; an execution failure (a dropped connection) is resumed from its stopping point, never rerun. Once an episode passes, its bulky records are compressed losslessly
(pi/api and pi/scientific_state/computations to .tar.zst, pi/pi_events.jsonl to .zst), each archive checked file by
file against the originals before they are removed. The grader reads none of those files."""
import hashlib
import json
import os
import random
import secrets
import signal
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PANEL = os.environ.get("SDL2_PANEL", "board")   # v1.3: board (C1) or heldout (C2)
assert PANEL in ("board", "heldout"), PANEL
QUEUE = HERE / ("QUEUE.json" if PANEL == "board" else "QUEUE_HELDOUT.json")
EPISODES = Path("/lab")   # v1.3: every split under /lab, random names (D12)
PY = str(STUDY / "python/bin/python")
STALL_S = 1800   # 30 minutes with no Pi event and no provider request or retry record (twice the native-command bound):
                 # a stalled provider stream, which the transport's inactivity timer misses because keep-alive bytes reset it
POLL_S = 60
STOP = threading.Event()   # set by Ctrl-C of `run`; a file named STOP in this folder also stops new starts and resumes
NODE = "/usr/bin/node"   # Node 24.18 on both Sparks; the PATH on spark-cfd0 finds a newer one first

# Jarrod, October 26: every arm once on all 96 sealed states (Qwen 3.8 Max and Grok 4.7 qualified but are not on the board).
MODELS = ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-astra", "google/gemini-4-argon",
          "deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3-flash"]   # v1.3: D14 adds GLM-5.3 Flash
ARMS = [{"arm": "frontier", "model": m} for m in MODELS] + [{"arm": "policy", "policy": "bo"},
                                                            {"arm": "policy", "policy": "doe"}]
# Queue order: the policies first (minutes each), then the longest episodes first, so slow arms never trail the run.
ORDER = ["bo", "doe", "anthropic/claude-opus-5.5", "google/gemini-4-argon", "deepseek/deepseek-v4.1-flash-1017",
         "z-ai/glm-5.3-flash",
         "openai/gpt-6-astra", "anthropic/claude-sonnet-5.5"]
# The context pilot's frozen decision for every model arm (context/PILOT_PREREG.md), e.g. {"native_command_timeout_s": 600}.
MODEL_FIELDS = {"native_command_timeout_s": 900}   # context/PILOT_DECISION.md: C2, the bound raised to 15 minutes


def label(run):
    return run.get("model") or run["policy"]


def build():
    if QUEUE.exists():
        sys.exit("QUEUE.json exists; it is built once")
    states = [st["name"] for st in json.loads((STUDY / f"owner/{PANEL.upper()}_PANEL.json").read_text())["states"]]
    seed = secrets.randbits(32)
    random.Random(seed).shuffle(states)
    entries = []
    arms = ARMS if PANEL == "board" else [{"arm": "frontier", "model": "deepseek/deepseek-v4.1-flash-1017"}]   # v1.3: C2
    for run in sorted(arms, key=lambda r: ORDER.index(label(r))):
        run = dict(run, **({} if run["arm"] == "policy" else MODEL_FIELDS))
        for k, state in enumerate(states):
            entries.append({"index": len(entries), "name": "e-" + secrets.token_hex(5), "state": state, "order": k,
                            "label": label(run), "run": run})
    # v1.3: the policies first, then each world's model arms together, longest first: the order v1.2's QUEUE.json ran
    # in after October 26, when an arm-blocked order saturated the CPU (v1.2 leaderboard/README.md, "Order").
    entries.sort(key=lambda e: (e["run"]["arm"] != "policy", 0 if e["run"]["arm"] == "policy" else e["order"],
                                ORDER.index(e["label"]), e["order"]))
    for i, e in enumerate(entries):
        e["index"] = i
    QUEUE.write_text(json.dumps({"panel": PANEL, "order_seed": seed, "entries": entries}, indent=1) + "\n")
    print(f"{len(entries)} entries: {len(states)} states x {len(arms)} arms")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack_dir(d):
    out = d.with_name(d.name + ".tar.zst")
    want = {str(p.relative_to(d.parent)): sha_file(p) for p in d.rglob("*") if p.is_file() and not p.is_symlink()}
    with open(out, "wb") as f:
        tar = subprocess.Popen(["tar", "-C", str(d.parent), "-cf", "-", d.name], stdout=subprocess.PIPE)
        subprocess.run(["zstd", "-q", "-3", "-T2", "-c"], stdin=tar.stdout, stdout=f, check=True)
        assert tar.wait() == 0
    got = {}
    unz = subprocess.Popen(["zstd", "-q", "-d", "-c", str(out)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=unz.stdout, mode="r|") as t:
        for m in t:
            if m.isfile():
                h, f = hashlib.sha256(), t.extractfile(m)
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
                got[m.name] = h.hexdigest()
            elif m.islnk():   # a second name of a file already archived
                got[m.name] = got[m.linkname]
    assert unz.wait() == 0 and got == want, f"archive check failed for {d}"
    subprocess.run(["rm", "-r", "--", str(d)], check=True)


def pack_file(p):
    out = p.with_name(p.name + ".zst")
    subprocess.run(["zstd", "-q", "-3", "-T2", str(p), "-o", str(out)], check=True)
    h, unz = hashlib.sha256(), subprocess.Popen(["zstd", "-q", "-d", "-c", str(out)], stdout=subprocess.PIPE)
    for b in iter(lambda: unz.stdout.read(1 << 20), b""):
        h.update(b)
    assert unz.wait() == 0 and h.hexdigest() == sha_file(p), f"archive check failed for {p}"
    p.unlink()


def stopping():
    return STOP.is_set() or (HERE / "STOP").exists()


def runner(e, *mode):
    """The runner's exit code, or "stalled" after interrupting a run with no activity for STALL_S seconds since it started:
    no new Pi event and no new provider request or retry record in pi/api."""
    ep = EPISODES / e["name"]
    marks, start = (ep / "pi/pi_events.jsonl", ep / "pi/api"), time.time()
    with open(ep / "run.log", "a") as log:
        p = subprocess.Popen([NODE, "run_v13.mjs", e["name"], *mode], cwd=STUDY / "env", stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "SDL2_V13_EPISODES": str(EPISODES)})
        while p.poll() is None:
            time.sleep(POLL_S)
            if time.time() - max([start] + [m.stat().st_mtime for m in marks if m.exists()]) > STALL_S:
                p.send_signal(signal.SIGINT)   # the runner saves LIVE-interrupted.json and removes its sandbox
                try:
                    p.wait(timeout=300)
                except subprocess.TimeoutExpired:   # stuck: kill it, and remove the sandbox its handler would have removed
                    p.kill()
                    p.wait()
                    subprocess.run(["docker", "rm", "-f", "ndfeb-sdl2v13-ep-" + e["name"]], capture_output=True)
                return "stalled"
        return p.returncode


def receipt(ep):
    """The newest receipt: a resume writes LIVE-resume.json, an interruption LIVE-interrupted.json."""
    found = [ep / f for f in ("LIVE-resume.json", "LIVE.json", "LIVE-interrupted.json", "POLICY.json") if (ep / f).exists()]
    return json.loads(max(found, key=lambda f: f.stat().st_mtime).read_text()) if found else None


def finish(e, code, seconds, resumes):
    ep = EPISODES / e["name"]
    r = receipt(ep)
    events, usage = ep / "pi/pi_events.jsonl", ep / "pi/usage.jsonl"
    calls = [json.loads(l) for l in usage.open()] if usage.exists() else []   # every attempt, resumes included
    acc = (r or {}).get("accounting") or {}
    cost = (sum(float((c.get("usage") or {}).get("cost") or 0) for c in calls) if calls
            else (acc.get("known_cost_usd") or 0) + (acc.get("estimated_cost_usd") or 0))
    summary = {"label": e["label"], "exit": code, "wall_seconds": round(seconds), "resumes": resumes,
               "status": (r or {}).get("status", "no receipt"), "termination": (r or {}).get("termination_cause"),
               "provider_calls": len(calls) if calls else (r or {}).get("provider_calls"),
               "compactions": sum(1 for l in events.open() if '"compaction_end"' in l) if events.exists() else 0,
               "cost_usd": round(cost, 4)}
    (ep / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    if summary["status"] == "pass":   # anything else is kept whole, so it can be resumed or diagnosed
        for d in (ep / "pi/api", ep / "pi/scientific_state/computations"):
            if d.is_dir():
                pack_dir(d)
        if events.exists():
            pack_file(events)
    print(f"{e['index']} {e['label']} {summary['status']} resumes {resumes} {time.strftime('%H:%M:%SZ', time.gmtime())}",
          flush=True)


def run_entry(e, panel):
    """Prepare, run, and resume an execution failure or a stall from its stopping point, up to twice."""
    ep = EPISODES / e["name"]
    if ep.exists() or stopping():
        return
    prep = subprocess.run([PY, "-B", "prepare_v13.py", e["name"], "--panel", panel, "--state", e["state"], "--root",
                           str(EPISODES)], cwd=STUDY / "env", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if prep.returncode:   # recorded, not stranded: nothing ran; move the folder aside and `run` starts it again
        ep.mkdir(parents=True, exist_ok=True)
        (ep / "SUMMARY.json").write_text(json.dumps({"label": e["label"], "status": "prepare_failed", "exit": prep.returncode,
                                                     "error": prep.stderr[-500:], "provider_calls": 0, "compactions": 0,
                                                     "cost_usd": 0}, indent=1) + "\n")
        return
    (ep / "RUN.json").write_text(json.dumps(e["run"]) + "\n")
    t0 = time.time()
    code = runner(e, "policy" if e["run"]["arm"] == "policy" else "live")
    resumes = 0
    while e["run"]["arm"] != "policy" and resumes < 2 and not stopping() and \
            (code == "stalled" or (receipt(ep) or {}).get("status") == "execution_failure"):
        time.sleep(60)
        if stopping():
            break
        resumes += 1
        code = runner(e, "live", "resume")
    finish(e, code, time.time() - t0, resumes)


def resume(name):
    """Resume one stopped model episode (interrupted, or still failed after the automatic resumes) from where it stopped."""
    e = next(e for e in json.loads(QUEUE.read_text())["entries"] if e["name"] == name)
    t0 = time.time()
    finish(e, runner(e, "live", "resume"), time.time() - t0, 1)


def run(n, shard="0/1"):
    # a launcher started in the background (nohup ... &) inherits SIGINT as ignored; restore it so a stop request works
    signal.signal(signal.SIGINT, signal.default_int_handler)
    k, s = map(int, shard.split("/"))
    q = json.loads(QUEUE.read_text())
    assert EPISODES.is_dir(), "/lab is the episode root (D12)"
    todo = [e for e in q["entries"] if e["order"] % s == k and not (EPISODES / e["name"]).exists()]
    pool = ThreadPoolExecutor(n)
    futures = [(e, pool.submit(run_entry, e, q["panel"])) for e in todo]
    try:
        for e, f in futures:
            try:
                f.result()
            except Exception as x:  # noqa: BLE001 - one failed entry is reported; the others run on
                print(f"{e['index']} {e['label']} ERROR {x!r}"[:300], flush=True)
    except KeyboardInterrupt:   # Ctrl-C reaches the runners too: each saves LIVE-interrupted.json; queued entries never start
        STOP.set()
        pool.shutdown(wait=True, cancel_futures=True)
        print("RUN-STOPPED", flush=True)
        raise SystemExit(130)
    pool.shutdown(wait=True)
    print("RUN-DONE", flush=True)


def status(shard="0/1"):
    k, s = map(int, shard.split("/"))
    rows = {}
    for e in json.loads(QUEUE.read_text())["entries"]:
        if e["order"] % s != k:
            continue
        ep, r = EPISODES / e["name"], rows.setdefault(e["label"], {"done": 0, "failed": 0, "running": 0, "pending": 0,
                                                                   "cost_usd": 0.0, "calls": 0, "compactions": 0})
        if (ep / "SUMMARY.json").exists():
            sm = json.loads((ep / "SUMMARY.json").read_text())
            r["done" if sm["status"] == "pass" else "failed"] += 1
            r["cost_usd"] += sm["cost_usd"]
            r["calls"] += sm["provider_calls"] or 0
            r["compactions"] += sm["compactions"]
        elif ep.exists():
            r["running"] += 1
            if (ep / "pi/usage.jsonl").exists():
                for l in (ep / "pi/usage.jsonl").open():
                    r["cost_usd"] += float((json.loads(l).get("usage") or {}).get("cost") or 0)
        else:
            r["pending"] += 1
    for k, r in rows.items():
        print(f"{k:36s} done {r['done']:3d} failed {r['failed']:2d} running {r['running']:2d} pending {r['pending']:3d}"
              f"  ${r['cost_usd']:8.2f}  calls {r['calls']:5d}  compactions {r['compactions']}")
    print(f"{'total':36s} ${sum(r['cost_usd'] for r in rows.values()):.2f}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"build": build, "status": lambda: status(*args), "resume": lambda: resume(args[0]),
     "run": lambda: run(int(args[0]), *args[1:])}[cmd]()
