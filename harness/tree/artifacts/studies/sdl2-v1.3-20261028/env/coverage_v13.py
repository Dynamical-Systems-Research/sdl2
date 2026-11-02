"""Refresh COVERAGE.md from the episode's public returns. Runs in the bridge environment; reads only what the agent's own
requests returned, from the runner's copy outside the agent-writable workspace. Usage: python -B coverage.py WORKSPACE RETURNS"""
import json
import os
import sys
import tempfile
from pathlib import Path

from ndfeb_sps import economy_v12 as economy

ws = Path(sys.argv[1])
source = Path(sys.argv[2])
returns = json.loads(source.read_text()) if source.exists() else []
state = economy.ledger([{"request_id": r["request_id"], "request": r["request"], "delivered": True} for r in returns])
text = economy.coverage_markdown(state)
# v1.3 (no hints): the agent's own spend, bodies and looped routes only; the list of declared values not yet built
# is an exploration checklist and is left out (env/port_env_text.py).
head, cut, tail = text.partition("\n## Declared values not yet built in any body\n")
if cut:
    text = head + "\n" + tail[tail.index("This table counts"):]
text += "\nEvery return so far, with observation IDs and values: `RETURNS.json`. The reference body: `INITIAL_EVIDENCE.json`.\n"
# The workspace is agent-writable: write a fresh file and rename it over the target, never following a planted link.
fd, tmp = tempfile.mkstemp(prefix=".sdl2-", dir=ws)
with os.fdopen(fd, "w") as f:
    f.write(text)
os.chmod(tmp, 0o644)
os.replace(tmp, ws / "COVERAGE.md")
