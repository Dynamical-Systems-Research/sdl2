"""v1.3.1 bridge: v1.2's frozen bridge environment copied into this study with one asserted edit (Jarrod, Oct 29).

The legacy laboratory (request_simulation without `simulation`) returned a bare `execution_failed`, while the magnetic
laboratory already returns the native error in `native_diagnostics`. Here the legacy laboratory reports its error the
same way, word for word. Only messages that match a known request-contract error pass through; any other text stays
owner-side, because a raw exception from the provider can carry owner paths or world-derived numbers. Nothing is
rewritten or added. Run once on each Spark (each copies its own v1.2 environment, so the interpreter link stays the
host's own):
  python3 -I env/port_bridge_v131.py"""
import shutil
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SRC = STUDY.parent / "sdl2-v1.2-20261026" / "bridge-python"
DST = STUDY / "bridge-python-v131"
assert not DST.exists(), f"{DST} exists; it is built once"
shutil.copytree(SRC, DST, symlinks=True)
f = DST / "lib/python3.13/site-packages/ndfeb_sps/investigation/hybrid_tools.py"
t = f.read_text()

A_IMPORT = "import copy\nimport hashlib\nimport json\n"
B_IMPORT = "import copy\nimport hashlib\nimport json\nimport re\n"

A_RETURN = """        delivery_id = result.get("delivery_id")
        if delivery_id:
            self.simulation.release(delivery_id, "native-conditional-simulation",
                                    call_id="virtual-release:" + delivery_id)
"""
B_RETURN = """        delivery_id = result.get("delivery_id")
        if not delivery_id:   # v1.3.1: the native error, word for word, as the magnetic laboratory reports it
            pending = next(row for row in self.simulation.state()["requests"] if row["request_id"] == rid)
            return {"status": "execution_failed", "request_id": rid, "delivery_id": None,
                    "model_id": committed["model_id"], "prediction": committed["prediction"],
                    "evidence_class": "conditional_computational", "origin": "synthetic",
                    "physical_execution": False, "scientific_negative": False,
                    "failure": pending["active_execution_failure"],
                    "native_diagnostics": _contract_errors(
                        self.simulation_root / "executions" / hashlib.sha256(rid.encode()).hexdigest()[:24]),
                    "artifacts": [],
                    "retrieval": "inspect_state(section='simulation') retains the failed request; corrected inputs require a new request ID"}
        if delivery_id:
            self.simulation.release(delivery_id, "native-conditional-simulation",
                                    call_id="virtual-release:" + delivery_id)
"""

A_HELPER = "class HybridInvestigator"
B_HELPER = '''# v1.3.1: request-contract errors the native simulator reports, passed to the investigator word for word. Any other
# text stays in the owner's receipt (a raw provider exception can name owner paths or world-derived values).
_CONTRACT_ERRORS = (
    re.compile(r"this conditional family implements only the accepted preparation; changed preparation needs new model support"),
    re.compile(r"density/loop acquisition requires measurement_context\\.field_set_id; retry the unchanged request for "
               r"redelivery, or choose a new ID for a fresh reading"),
    re.compile(r"[A-Za-z0-9_.-]{1,64}: declare preparation '[A-Za-z0-9_.-]{1,64}' with its conditions in "
               r"request\\.preparations \\(identical to its first declaration\\)"),
)


def _contract_errors(artifacts):
    out = []
    for receipt in sorted(Path(artifacts).glob("native/native-*/run_receipt.json")):
        try:
            reasons = json.loads(receipt.read_text()).get("validation_reasons") or []
        except (OSError, ValueError):
            continue
        for r in reasons:
            detail = r.get("detail") if isinstance(r, dict) else None
            if r.get("code") == "SPS_RESPONSE_UNAVAILABLE" and isinstance(detail, str) \\
                    and any(p.fullmatch(detail) for p in _CONTRACT_ERRORS):
                out.append({"code": "SPS_RESPONSE_UNAVAILABLE", "detail": detail})
    return out


class HybridInvestigator'''

for a, b in ((A_IMPORT, B_IMPORT), (A_RETURN, B_RETURN), (A_HELPER, B_HELPER)):
    assert t.count(a) == 1, (a[:60], t.count(a))
    t = t.replace(a, b)
f.write_text(t)
print("built", DST, "and edited", f.relative_to(STUDY))
