#!/usr/bin/env bash
# Run one leaderboard world end to end with the harness assembled by setup.sh.
#   harness/run.sh ROOT EPISODES NAME STATE bo|doe    a classical baseline, no model calls
#   harness/run.sh ROOT EPISODES NAME STATE MODEL     a language model through OpenRouter (paid; needs OPENROUTER_API_KEY)
# NAME is the episode directory (e-<10 hex>); a baseline's seed derives from it, so reusing a recorded name reproduces
# that episode. STATE is board-001 to board-032. MODEL is one of the OpenRouter IDs in the README. No directory above
# EPISODES may contain .git.
set -euo pipefail
ROOT=$(realpath "$1"); EPISODES=$(realpath -m "$2"); NAME=$3; STATE=$4; ARM=$5
EVAL=$ROOT/artifacts/studies/evaluation
mkdir -p "$EPISODES"
(cd "$EVAL/env" && "$EVAL/python/bin/python" -B prepare.py "$NAME" --panel board --state "$STATE" --root "$EPISODES" >/dev/null)
case $ARM in
  bo|doe) printf '{"arm": "policy", "policy": "%s"}\n' "$ARM" > "$EPISODES/$NAME/RUN.json"; MODE=policy ;;
  *) [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }
     printf '{"arm": "frontier", "model": "%s", "native_command_timeout_s": 900}\n' "$ARM" > "$EPISODES/$NAME/RUN.json"; MODE=live ;;
esac
cd "$EVAL/env" && SDL2_EPISODES=$EPISODES node run.mjs "$NAME" "$MODE"
