"""Readers for every record an SDL-2 episode keeps, raw or compressed (artifacts/proposals/sdl2-release-20261026/
TRACE_MINING_PLAN.md). Owner only. A passed leaderboard episode has pi/pi_events.jsonl.zst, pi/api.tar.zst and
pi/scientific_state/computations.tar.zst in place of the raw files; every reader here takes either form, so no
episode is silently read as empty."""
import json
import subprocess
import tarfile
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RECEIPTS = ("LIVE-resume.json", "LIVE.json", "LIVE-interrupted.json", "POLICY.json", "E2E.json")


def _jsonl_lines(raw, zst):
    if raw.exists():
        with raw.open() as f:
            yield from f
    elif zst.exists():
        p = subprocess.Popen(["zstd", "-q", "-d", "-c", str(zst)], stdout=subprocess.PIPE, text=True)
        yield from p.stdout
        assert p.wait() == 0, f"zstd failed on {zst}"


def events(ep):
    """Pi events in order: {"utc", "event": {...}}."""
    ep = Path(ep)
    for line in _jsonl_lines(ep / "pi/pi_events.jsonl", ep / "pi/pi_events.jsonl.zst"):
        if line.strip():
            yield json.loads(line)


def session(ep):
    """The full Pi transcript (never compressed): every entry of the episode's session file(s), in order."""
    out = []
    for f in sorted((Path(ep) / "pi/pi_sessions").glob("*.jsonl")):
        out += [json.loads(line) for line in f.open() if line.strip()]
    return out


def calls(ep):
    """Every scientific tool call through the bridge, with its arguments and result."""
    f = Path(ep) / "calls.jsonl"
    return [json.loads(line) for line in f.open() if line.strip()] if f.exists() else []


def usage(ep):
    f = Path(ep) / "pi/usage.jsonl"
    return [json.loads(line) for line in f.open() if line.strip()] if f.exists() else []


def api_files(ep, prefix="response_"):
    """(name, bytes) for each raw provider record in pi/api, raw or from pi/api.tar.zst."""
    ep = Path(ep)
    d = ep / "pi/api"
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.name.startswith(prefix):
                yield f.name, f.read_bytes()
    elif (ep / "pi/api.tar.zst").exists():
        p = subprocess.Popen(["zstd", "-q", "-d", "-c", str(ep / "pi/api.tar.zst")], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=p.stdout, mode="r|") as t:
            for m in t:
                name = m.name.split("/")[-1]
                if m.isfile() and name.startswith(prefix):
                    yield name, t.extractfile(m).read()
        assert p.wait() == 0


def receipt(ep):
    """The newest receipt (a resume writes LIVE-resume.json, an interruption LIVE-interrupted.json)."""
    found = [Path(ep) / f for f in RECEIPTS if (Path(ep) / f).exists()]
    return json.loads(max(found, key=lambda f: f.stat().st_mtime).read_text()) if found else None


def identity(ep):
    """Arm label and world identity of an episode, from its own records (works for any root)."""
    ep = Path(ep)
    run = json.loads((ep / "RUN.json").read_text()) if (ep / "RUN.json").exists() else {}
    label = run.get("model") or run.get("policy") or ("deepseek-1017-tinker" if run.get("arm", "tinker-1017") == "tinker-1017" else run.get("arm"))
    origin = json.loads((ep / "PREPARATION.json").read_text()).get("origin", {})
    return {"episode": ep.name, "label": label, "arm": run.get("arm", "tinker-1017"), "panel": origin.get("panel"),
            "state": origin.get("state"), "seed": origin.get("seed")}
