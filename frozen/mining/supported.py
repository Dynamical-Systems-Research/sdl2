"""v1.3 Justified for the report: Supported (grader.2, grader/RUNG2_DECISION.md) from GRADES.jsonl, in the rows format
report.py reads for Justified. Owner only. Writes, next to GRADES.jsonl:
  SUPPORTED.jsonl  one row per graded episode and K: episode, standard (K30, K100, K200), justified (Supported at that K;
                   none and no claim are never Supported; an ungradable record is null and the report counts it pending)
  C2S.jsonl        one row per episode with a restrict or nominate claim: episode, credits_to_supported, lost
Then: report.py OUT ... --justified OUT/SUPPORTED.jsonl --declared K100 --credits-to-justified OUT/C2S.jsonl
(K = 100 is pre-registered; K30 and K200 fill the report's sensitivity table).
  ../python/bin/python -B supported.py OUT_DIR"""
import json
import sys
from pathlib import Path


def main(out):
    out = Path(out)
    rows = [json.loads(line) for line in (out / "GRADES.jsonl").open()]
    with open(out / "SUPPORTED.jsonl", "w") as fs, open(out / "C2S.jsonl", "w") as fc:
        for g in rows:
            for k, v in (g.get("supported_by_K") or {"30": None, "100": None, "200": None}).items():
                fs.write(json.dumps({"episode": g["episode"], "standard": f"K{k}", "justified": v}) + "\n")
            c = g.get("credits_to_supported")
            if c is not None:
                fc.write(json.dumps({"episode": g["episode"], "credits_to_supported": c["credits_to_supported"],
                                     "lost": c["lost"]}) + "\n")
    print(f"{len(rows)} episodes -> SUPPORTED.jsonl, C2S.jsonl")


if __name__ == "__main__":
    main(sys.argv[1])
