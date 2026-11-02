"""Build the v1.3 classical policy arms from v1.2's (not edited) with exact-text edits, each required to match the stated
number of times, and record the diffs (A6). The protocol is v1.2's arms/POLICY_PROTOCOL.md unchanged, except:
  - the bar: a route meets when p3 >= 0.95 (CONTRACT.md), in the policy and in the state's public requirement;
  - the prior: the supplied C7.3 model's ranking recomputed on the shipped split (worlds/ranking/, 8,064 routes).
The density reading's sd field is read as before; in v1.3 it is the method's nominal sd (CONTRACT.md "Instruments").

  python3 arms/port_policies.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
V12 = HERE.parent.parent / "sdl2-v1.2-20261026/arms"

JOBS = {
    "classical/policy_v13.py": ("classical/policy_v12.py", [
        ('"""SDL-2 v1.2 classical policy arms:', '"""SDL-2 v1.3 classical policy arms (built from ../sdl2-v1.2-20261026/arms/classical/'
         'policy_v12.py\nby arms/port_policies.py; diff in arms/classical/policy_v13.diff):', 1),
        ("python/bin/python -B arms/classical/policy_v12.py STATE.json", "python/bin/python -B arms/classical/policy_v13.py STATE.json", 1),
        ("p3 >= 0.5", "p3 >= 0.95", 2),
        ('REQUIREMENT = {"hci_kOe": 10.0, "bhmax_MGOe": 11.0, "bodies": 3, "p_all": 0.5}',
         'REQUIREMENT = {"hci_kOe": 10.0, "bhmax_MGOe": 11.0, "bodies": 3, "p_all": 0.95}', 1),
    ]),
    "policy_state_v13.py": ("policy_state.py", [
        ('"""Build a policy arm\'s STATE.json', '"""(v1.3; built from ../sdl2-v1.2-20261026/arms/policy_state.py by arms/port_policies.py.)\n'
         'Build a policy arm\'s STATE.json', 1),
        ('"p_all": 0.5, "measurement_temperature_C": 20.0}', '"p_all": 0.95, "measurement_temperature_C": 20.0}', 1),
        ('"prior": str(STUDY / "arms/prior/supplied_ranking.jsonl")}', '"prior": str(STUDY / "worlds/ranking")}', 1),
    ]),
}


def main():
    for dst, (src, edits) in JOBS.items():
        old = (V12 / src).read_text()
        text = old
        for a, b, n in edits:
            assert text.count(a) == n, (dst, a[:80], text.count(a), n)
            text = text.replace(a, b)
        (HERE / dst).parent.mkdir(parents=True, exist_ok=True)
        (HERE / dst).write_text(text)
        diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True), f"sdl2-v1.2-20261026/arms/{src}",
                                    f"sdl2-v1.3-20261028/arms/{dst}")
        (HERE / dst).with_suffix(".diff").write_text("".join(diff))
        print(f"{len(edits)} edits -> arms/{dst}")


if __name__ == "__main__":
    main()
