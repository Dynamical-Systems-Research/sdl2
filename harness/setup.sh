#!/usr/bin/env bash
# Assemble the frozen harness under ROOT from this repository and the dataset (DATA).
#   harness/setup.sh ROOT DATA
# Needs Linux aarch64, Docker with the NVIDIA container runtime, Node 24, uv, host UID 1000, and ripgrep 15.2.0 and
# fd 10.4.2 in ~/.pi/agent/bin. No directory above ROOT may contain .git.
set -euo pipefail
ROOT=$(realpath -m "$1"); DATA=$(realpath "$2"); HERE=$(cd "$(dirname "$0")" && pwd)
EVAL=$ROOT/artifacts/studies/evaluation
BRIDGE=$ROOT/artifacts/studies/bridge
PI=$ROOT/artifacts/studies/agent-runtime/runtime/pi

d=$ROOT; while [ "$d" != / ]; do [ -e "$d/.git" ] && { echo "ROOT has a .git above it ($d); the bridge refuses that"; exit 1; }; d=$(dirname "$d"); done
mkdir -p "$ROOT"
rsync -a "$HERE/tree/" "$ROOT/"

# Environment data from the dataset (the dataset card lists what is withheld)
mkdir -p "$EVAL/env" "$EVAL/worlds" "$BRIDGE/env" "$ROOT/artifacts/studies/vision-campaign/sources"
rsync -a "$DATA/environment/library-agent/" "$EVAL/env/library-agent/"
rsync -a "$DATA/environment/models-agent/" "$EVAL/env/models-agent/"
rsync -a "$DATA/environment/base-investigation/" "$BRIDGE/env/base-investigation/"
rsync -a "$DATA/environment/historical-images/" "$ROOT/artifacts/studies/vision-campaign/sources/historical-images/"
rsync -a "$DATA/worlds/ranking/" "$EVAL/worlds/ranking/"
mkdir -p "$EVAL/owner"
rsync -a "$DATA/worlds/board-package/" "$EVAL/owner/package-world-eval/"
python3 - "$DATA/worlds/board_panel.json" "$EVAL/owner/BOARD_PANEL.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
json.dump({"states": [{k: s[k] for k in ("name", "world", "seed", "class", "feasible")} for s in p["states"]]},
          open(sys.argv[2], "w"), indent=1)
EOF
# Point the base investigation's absolute study paths at ROOT
grep -rlZ --include='*.json' '/home/jarrodbarnes/ndfeb-sps-campaign' "$BRIDGE/env/base-investigation" \
  "$ROOT/artifacts/studies/material-lab/template" | xargs -0 -r sed -i "s#/home/jarrodbarnes/ndfeb-sps-campaign#$ROOT#g"

# Both native runtimes are dynamical-cli at its pinned public commit plus the six files this repository ships for each
# (the ndfeb-sps adapter, its dependency lock and bundle, and two registration files). The bridge requires each runtime
# to be a git checkout of that commit.
COMMIT=b4823fb4e4babd942e3beafcad8042ea0684f1c0
for rt in "$EVAL/python/native-runtime" "$ROOT/.runtime/legacy-lab-runtime"; do
  if [ ! -d "$rt/.git" ]; then
    git clone -q https://github.com/Dynamical-Systems-Research/dynamical-cli "$rt.git-tmp"
    git -C "$rt.git-tmp" checkout -q "$COMMIT"
    rsync -a "$rt/" "$rt.git-tmp/"
    rm -rf "$rt" && mv "$rt.git-tmp" "$rt"
  fi
done

# Python environments: exact pins, the vendored ndfeb_sps trees and the leaderboard package route. run.mjs hands the
# bridge interpreter to Pi as NDFEB_PYTHON; run_verbatim_errors.mjs uses the verbatim-errors bridge instead.
for env in "$EVAL/python:$EVAL/vendor/owner-site-packages" "$BRIDGE/bridge-python:$BRIDGE/vendor/bridge-site-packages" \
           "$EVAL/bridge-python-verbatim-errors:$EVAL/vendor/bridge-verbatim-errors-site-packages"; do
  dir=${env%%:*}; vendor=${env##*:}
  uv venv -q --python 3.13.12 --allow-existing "$dir"
  VIRTUAL_ENV=$dir uv pip install -q -r "$HERE/requirements.txt"
  site=$("$dir/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  rsync -a "$vendor/" "$site/"
  cp "$HERE/sitecustomize.py" "$site/sitecustomize.py"
  "$dir/bin/python" -S - "$site" "$EVAL/owner/package-world-eval" "$DATA/worlds/board-package/FROZEN_PACKAGE_SHA256.json" <<'EOF'
import hashlib, json, sys
from pathlib import Path
site, package, frozen = sys.argv[1], Path(sys.argv[2]), json.load(open(sys.argv[3]))
files = {"tool": "ndfeb_physics_tool.py", "physics": "physics.py", "forward": "forward.py", "posterior": "posterior_tool.npz",
         "microstructure": "microstructure.py", "coercivity_link": "coercivity_link.py", "anneal": "anneal.py",
         "anneal_posterior": "anneal_posterior.npz"}
sha = {k: hashlib.sha256((package / v).read_bytes()).hexdigest() for k, v in files.items() if k in frozen}
json.dump({"package_dir": str(package), "frozen_sha256": frozen, "package_sha256": sha}, open(Path(site) / "sdl2_package.json", "w"), indent=1)
EOF
done

# Pi 1.4.0 from its frozen lockfile
(cd "$PI" && npm ci --silent --no-audit --no-fund)

# Images, loaded by the IDs the frozen runner names
for img in "sha256:1b1f2036d24ec4f78ecca6aedd0c956ba903342e04d70e3ed712da37ec4ae34a:sandbox" \
           "sha256:1bbaab57e6430f48f6d709b1197781a9e9458b8781c8f926e18bef05df69fc05:science"; do
  id=${img%:*}; name=${img##*:}
  docker image inspect "$id" >/dev/null 2>&1 || zstd -dc "$DATA/images/$name.tar.zst" | docker load
  docker image inspect "$id" >/dev/null || { echo "image $id not available"; exit 1; }
done
for t in rg fd; do [ -x "$HOME/.pi/agent/bin/$t" ] || echo "warning: $HOME/.pi/agent/bin/$t missing (Pi mounts it into the sandbox)"; done
echo "SDL-2 harness ready under $ROOT"
