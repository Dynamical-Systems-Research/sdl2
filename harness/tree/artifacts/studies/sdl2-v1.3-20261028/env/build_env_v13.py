"""Build this study's owner environment the way v1.2's env/build_env.py built its own: copy v1.2's owner environment
(which already carries sdl2_v12 and economy_v12), add the v1.3 world module, one native dispatch branch and its
dependency-lock entry, and re-bind the adapter and lock hashes in facility and registry. v1.2's environment and the
live environments are not touched. The bridge environment is v1.2's, used read-only (the economy is unchanged).

  ../sdl2-v1.2-20261026/python/bin/python -B env/build_env_v13.py
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
V12 = STUDY.parent / "sdl2-v1.2-20261026"
OWNER_ENV = STUDY / "python"
SITE = Path("lib/python3.13/site-packages")
NATIVE = Path("native-runtime/dynamical")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    if not OWNER_ENV.exists():
        shutil.copytree(V12 / "python", OWNER_ENV, symlinks=True)
    # The world module, used by the adapter for simulate_sdl2_v13 only. Owner-side draws (prepare_v13.py, the grader)
    # import src/sdl2_v13.py, whose STUDY path resolves to this study's owner/ directory.
    module = OWNER_ENV / SITE / "ndfeb_sps/models/sdl2_v13.py"
    shutil.copy2(STUDY / "src/sdl2_v13.py", module)
    adapter = OWNER_ENV / NATIVE / "instruments/ndfeb_sps.py"
    lockpath = OWNER_ENV / NATIVE / "bundle/ndfeb-sps/dependency_lock.json"
    old_adapter, old_lock = sha(adapter), sha(lockpath)
    lock = json.loads(lockpath.read_text())
    locked = {"models/sdl2_v13.py": sha(module)}
    lock["dependencies"] = [d for d in lock["dependencies"] if d["path"] not in locked]
    lock["dependencies"] += [{"path": k, "sha256": v} for k, v in locked.items()]
    lockpath.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    text = adapter.read_text()
    anchor = '        elif family == "ndfeb_c7_campaign_v1":'
    branch = ('        elif family == "ndfeb_sdl2_v13":\n'
              '            from ndfeb_sps.models.sdl2_v13 import simulate_sdl2_v13\n'
              '            delivery = simulate_sdl2_v13(context["request"], world_config=world, seed=context["seed"])\n')
    if branch not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, branch + anchor)
    text = text.replace(old_lock, sha(lockpath))
    adapter.write_text(text)
    for name in ("facility.yaml", "registry.yaml"):
        p = OWNER_ENV / NATIVE / "bundle/ndfeb-sps" / name
        p.write_text(p.read_text().replace(old_adapter, sha(adapter)).replace(old_lock, sha(lockpath)))
    receipt = {"owner_environment": "python (copied from ../sdl2-v1.2-20261026/python)",
               "bridge_environment": "../sdl2-v1.2-20261026/bridge-python (read-only)",
               "module_sha256": sha(module), "src_sdl2_v13_sha256": sha(STUDY / "src/sdl2_v13.py"),
               "adapter_sha256": sha(adapter), "dependency_lock_sha256": sha(lockpath),
               "sdl2_v12_sha256": sha(OWNER_ENV / SITE / "ndfeb_sps/models/sdl2_v12.py"),
               "c7_campaign_sha256": sha(OWNER_ENV / SITE / "ndfeb_sps/models/c7_campaign.py"),
               "economy_v12_sha256": sha(OWNER_ENV / SITE / "ndfeb_sps/economy_v12.py")}
    (HERE / "BUILD_V13.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
