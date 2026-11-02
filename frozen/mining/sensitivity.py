"""The thin-margin sensitivity (TRACE_MINING_PLAN.md, Oct 29; worlds/thin_margin.py defines the flag). Owner only.
For each arm: Both and material outcome on all worlds (the headline, unchanged), and on the worlds that are not
thin-margin, with exact 95% intervals. On the thin-margin worlds it also records the claims made and their Outcome,
descriptively: there a correct claim reflects the decision under an unresolvable margin, not the search. It never
reranks the board. Writes SENSITIVITY.json and SENSITIVITY.md next to GRADES.jsonl.
  ../python/bin/python -B sensitivity.py OUT_DIR"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report import ORDER, SHORT, rate  # noqa: E402

THIN = json.loads((HERE.parent / "owner/THIN_MARGIN.json").read_text())["panels"]


def thin(g):
    return bool(THIN.get(f"{(g.get('panel') or '').upper()}_PANEL", {}).get(g.get("state"), False))


def main(out):
    out = Path(out)
    rows = [g for g in map(json.loads, (out / "GRADES.jsonl").open()) if "grade_error" not in g and g.get("panel")]
    labels = [a for a in ORDER if any(g["label"] == a for g in rows)] + sorted({g["label"] for g in rows} - set(ORDER))
    res = {"rule": "thin-margin feasible world: no passing route's median body reads P(pass) >= 0.95 (worlds/thin_margin.py)",
           "worlds": {"all": len({g["world_key"] for g in rows}), "thin": len({g["world_key"] for g in rows if thin(g)})},
           "arms": {}}
    for a in labels:
        fs = [g for g in rows if g["label"] == a]
        keep = [g for g in fs if not thin(g)]
        on_thin = [g for g in fs if thin(g)]
        res["arms"][a] = {
            "all": {"Both": rate(g["Both"] for g in fs),
                    "material_outcome": rate(g["Outcome"] for g in fs if g["world"]["feasible"])},
            "without_thin": {"Both": rate(g["Both"] for g in keep),
                             "material_outcome": rate(g["Outcome"] for g in keep if g["world"]["feasible"])},
            "on_thin_observed": {"claims": dict(Counter(g["claim_form"] for g in on_thin)),
                                 "Outcome": sum(bool(g["Outcome"]) for g in on_thin), "Both": sum(bool(g["Both"]) for g in on_thin),
                                 "n": len(on_thin)}}
    (out / "SENSITIVITY.json").write_text(json.dumps(res, indent=1) + "\n")
    f = lambda r: f"{r['k']}/{r['n']}" + (f" [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]" if r["n"] else "")  # noqa: E731
    L = ["# Thin-margin sensitivity (never reranks the board)", "",
         f"{res['worlds']['thin']} of {res['worlds']['all']} worlds are thin-margin feasible worlds ({res['rule']}).", "",
         "| Arm | Both, all worlds | Both, without thin | Material outcome, all feasible | Material outcome, without thin | Claims on thin worlds (Outcome) |",
         "|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        t = r["on_thin_observed"]
        L.append(f"| {SHORT.get(a, a)} | {f(r['all']['Both'])} | {f(r['without_thin']['Both'])} | "
                 f"{f(r['all']['material_outcome'])} | {f(r['without_thin']['material_outcome'])} | "
                 f"{', '.join(f'{k} {v}' for k, v in sorted(t['claims'].items())) or '-'} ({t['Outcome']}/{t['n']}) |")
    (out / "SENSITIVITY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
