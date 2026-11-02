"""A9: write FROZEN_V13.json, the hash of every file that defines the v1.3 environment, contract, worlds, arms, grader
and mining, and of the inputs outside this study that an episode reads (built from ../sdl2-v1.2-20261026/freeze.py).
A sealed or training run records this file's hash; any later change makes a new build. Run on each Spark and compare
(everything but host, label and utc must be identical):
  python/bin/python -B freeze_v13.py LABEL [OUT.json]"""
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
V12 = HERE.parent / "sdl2-v1.2-20261026"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(root):
    h = hashlib.sha256()
    n = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "node_modules"))
        for f in sorted(x for x in files if not x.endswith(("-shm", "-wal", ".pyc"))):
            p = Path(dirpath) / f
            h.update(str(p.relative_to(root)).encode() + b"\0" + sha(p).encode() + b"\n")
            n += 1
    return {"files": n, "sha256": h.hexdigest()}


FILES = [  # the agent's world and laboratory
    "src/sdl2_v13.py", "env/CONTRACT.md", "env/LABORATORY.md", "env/prepare_v13.py", "env/run_v13.mjs",
    "env/coverage_v13.py", "env/build_env_v13.py", "env/BUILD_V13.json", "env/port_env_text.py", "env/port_prepare.py",
    "env/port_runner.py", "env/TEXT_V13.diff", "env/prepare_v13.diff", "env/run_v13.diff",
    "build/split_packages.py", "build/world_packages.py", "build/fresh_anneal.py", "build/SPLIT_RECEIPT.json",
    "build/WORLD_PACKAGES.json", "build/NOMINAL_SD.json",
    # worlds and panels (owner only), drawn under custody
    "worlds/strata_v13.py", "worlds/dev_panel_v13.py", "worlds/a5b_candidates.py", "worlds/a5b_select.py",
    "worlds/supplied_ranking_v13.py", "owner/WORLD_SPLIT.json", "owner/ANNEAL_SEEDS.json", "owner/anneal_world_eval.npz",
    "owner/anneal_world_train.npz", "owner/DEV_PANEL_CUSTODY.json", "owner/DEV_PANEL.json", "owner/A5B_CUSTODY.json",
    "owner/A5B_CANDIDATES.jsonl", "owner/A5B_SELECTION.json", "owner/BOARD_PANEL.json", "owner/HELDOUT_PANEL.json",
    # arms
    "arms/classical/policy_v13.py", "arms/classical/policy_v13.diff", "arms/policy_state_v13.py",
    "arms/policy_state_v13.diff", "arms/port_policies.py", "arms/frontier/openrouter_transport.mjs",
    "arms/frontier/port_transport.py", "arms/frontier/openrouter_transport.diff", "arms/B_PILOT.json",
    "arms/b_pilot_launch.sh",
    # grader: rung 1 and Supported (grader.2); the rung-2 observer is a diagnostic
    "grader/grade_v13.py", "grader/port_grade.py", "grader/grade_v13.diff", "grader/truth_v13.py",
    "grader/justify_v13.py", "grader/justify_tools_v13.py", "grader/port_justify.py", "grader/justify_v13.diff",
    "owner/pool_grader.jsonl",
    # mining and its pre-registration
    "mining/port_mining.py", "mining/MINING_V13.diff", "mining/grade_all.py", "mining/features.py", "mining/report.py",
    "mining/supported.py", "mining/traces.py", "mining/excerpts.py", "mining/replays.py", "mining/check_replays.py",
    "mining/locators.py", "TRACE_MINING_PLAN.md", "MEASUREMENT_SPEC.md", "grader/CALIBRATION_V13.md",
    "grader/RUNG2_DECISION.md", "grader/rung2_evidence/RESTRICTION_SWEEP.json",
    "worlds/thin_margin.py", "owner/THIN_MARGIN.json", "mining/sensitivity.py",
    # Phase C (Jarrod's go, Oct 29): the launcher, its procedure and both queues, built once
    "leaderboard/leaderboard_v13.py", "leaderboard/port_leaderboard.py", "leaderboard/leaderboard_v13.diff",
    "leaderboard/README.md", "leaderboard/QUEUE.json", "leaderboard/QUEUE_HELDOUT.json",
    # checks rerun at the freeze
    "checks/leak_check_v13.py", "checks/leak_bridge_v13.py", "checks/a9_checks.sh", "sync_cfd0.sh", "freeze_v13.py"]
TREES = ("env/library-agent", "env/models-agent", "owner/package-world-eval", "owner/package-world-train",
         "worlds/ranking", "python/lib/python3.13/site-packages/ndfeb_sps", "python/native-runtime")
OUTSIDE_FILES = {   # read-only inputs outside the study (v1.2 is frozen; its FROZEN.json covers the rest of it)
    "v12/FROZEN.json": V12 / "FROZEN.json",
    "v12/route/ds_tinker_transport.mjs": V12 / "route/ds_tinker_transport.mjs",
    "v12/route/ds_tinker_worker.py": V12 / "route/ds_tinker_worker.py",
    "v12/arms/frontier/openrouter_transport.mjs": V12 / "arms/frontier/openrouter_transport.mjs",
    "material-lab/owner_simulation_runtime.json":
        HERE.parent / "material-lab-20261019/template/investigation/simulation-magnetic/owner_simulation_runtime.json"}
OUTSIDE_TREES = {"v12/bridge-python/ndfeb_sps": V12 / "bridge-python/lib/python3.13/site-packages/ndfeb_sps",
                 "v12/env/base-investigation": V12 / "env/base-investigation",
                 "material-lab/instruments": HERE.parent / "material-lab-20261019/instruments"}
PI140 = HERE.parent / "material-route-swarm-design-20261019/runtime/pi140"
PROVIDERS = [HERE.parent / "physical-fresh-20261012/runtime/providers", V12 / "arms/frontier/providers",
             HERE.parent / "engine-release-20261008/v2-build/providers-gemini4-argon"]
IMAGE = "sha256:1bbaab57e6430f48f6d709b1197781a9e9458b8781c8f926e18bef05df69fc05"   # env/run_v13.mjs NDFEB_SCIENCE_IMAGE


def providers():
    """Every frontier PIN with the catalog and endpoint snapshots it names (they set each route's context window)."""
    repo = HERE.parents[2]
    out = {}
    for pin in sorted(p for d in PROVIDERS for p in d.glob("*__PIN.json")):
        out[pin.name] = sha(pin)
        rec = json.loads(pin.read_text())
        for key in ("catalog_file", "endpoints_file"):
            if rec.get(key):
                out[rec[key]] = sha(repo / rec[key])
    return out


if __name__ == "__main__":
    image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], capture_output=True, text=True)
    out = {"label": sys.argv[1] if len(sys.argv) > 1 else "build", "host": socket.gethostname(),
           "utc": datetime.now(timezone.utc).isoformat(),
           "files": {f: sha(HERE / f) for f in FILES},
           "routes": len(__import__("ndfeb_sps.economy_v12", fromlist=["x"]).declared_routes()),
           "trees": {t: tree(HERE / t) for t in TREES},
           "outside": {**{k: sha(v) for k, v in OUTSIDE_FILES.items()}, **{k: tree(v) for k, v in OUTSIDE_TREES.items()}},
           "providers": providers(),
           "pi140": {f: sha(PI140 / f) for f in ("vision_transport.mjs", "launch_vision.mjs", "native_sandbox.mjs",
                                                  "package-lock.json")},
           "sandbox_image": image.stdout.strip() if image.returncode == 0 else None,
           "node": subprocess.run(["/usr/bin/node", "--version"], capture_output=True, text=True).stdout.strip(),
           "python": sys.version.split()[0], "training": False}
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "FROZEN_V13.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"label": out["label"], "host": out["host"], "files": len(out["files"]),
                      "sha256_without_host_label_utc": hashlib.sha256(json.dumps(
                          {k: v for k, v in out.items() if k not in ("label", "host", "utc")}, sort_keys=True).encode()
                      ).hexdigest()}, indent=1))
