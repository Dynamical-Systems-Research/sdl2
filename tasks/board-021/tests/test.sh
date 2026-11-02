#!/usr/bin/env bash
# Grade one finished episode with the frozen grader; writes EPISODE/GRADE.json and EPISODE/reward.json
# ({"outcome": 0|1, "supported": 0|1, "both": 0|1}). Copies reward.json to /logs/verifier/ when that directory exists.
set -euo pipefail
DATA=$1; EPISODE=$2
cd "$(dirname "$0")/../../.."
uv run sdl2 grade "$DATA" "$EPISODE"
[ -d /logs/verifier ] && cp "$EPISODE/reward.json" /logs/verifier/reward.json || true
