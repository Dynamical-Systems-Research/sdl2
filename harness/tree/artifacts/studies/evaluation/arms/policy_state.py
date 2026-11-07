"""Build a policy arm's STATE.json from what an agent sees: the contract's public numbers,
the reference body, every executed request with its public delivery, the ledger those imply, the policy's own action
history (stdin) and the path of the disclosed prior. Reads no owner file.
Usage: python -B policy_state.py EPISODE_DIR ARM SEED OUT_JSON < history.json
"""
import json
import sys
from pathlib import Path

from ndfeb_sps import economy as economy

STUDY = Path(__file__).resolve().parents[1]
# CONTRACT.md "Requirement" (public)
REQUIREMENT = {"hci_kOe": 10.0, "bhmax_MGOe": 11.0, "bodies": 3, "p_all": 0.95, "measurement_temperature_C": 20.0}

ep, arm, seed, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
history = json.loads(sys.stdin.read() or "[]")
returns = json.loads((ep / "RETURNS.owner.json").read_text())   # the runner's copy of the agent's RETURNS.json
reference = json.loads((ep / "pi/workspace/INITIAL_EVIDENCE.json").read_text())["observations"]
led = economy.ledger([{"request_id": r["request_id"], "request": r["request"], "delivered": True} for r in returns])
state = {"arm": arm, "seed": seed,
         "contract": {"allowance": economy.ALLOWANCE, "prices": economy.PRICES, "route_keys": list(economy.ROUTE_KEYS),
                      "declared": {k: list(v) for k, v in economy.DECLARED.items()},
                      "anneal_programs": {k: [list(s) for s in v] for k, v in economy.ANNEAL_PROGRAMS.items()},
                      "fixed": economy.FIXED, "requirement": REQUIREMENT, "lot_id": "SDL2-LOT"},
         "reference": reference, "returns": returns,
         "ledger": {k: led[k] for k in ("credits", "new_bodies", "counts", "within_allowance")},
         "history": history, "prior": str(STUDY / "worlds/ranking")}
out.write_text(json.dumps(state, default=str) + "\n")
