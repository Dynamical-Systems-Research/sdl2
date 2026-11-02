#!/usr/bin/env python3
"""A2: the agent-visible text for v1.3, built from v1.2's env/CONTRACT.md and env/LABORATORY.md (unchanged) by exact
edits that must each match once. Writes env/CONTRACT.md, env/LABORATORY.md and env/TEXT_V13.diff.

Edits, each tied to a decision: D1 the 0.95 bar; the structural fix (worlds are not stored rows); the v1.3 readings
(nominal density sd, knee, SEM width, SPS log); D5 (the reference body is not re-operated); D10 (no network; the library);
no hints (Jarrod, Oct 28): the agent is told the decision problem, the rules and the instruments, and the base rate the
rung-2 ideal observer also uses, and nothing about panel selection or method. COVERAGE.md (env/coverage_v13.py from
v1.2's env/coverage.py) keeps the agent's own spend, bodies and looped routes and drops the list of declared values not
yet built, an exploration checklist.
  python3 env/port_env_text.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
V12 = HERE.parents[1] / "sdl2-v1.2-20261026/env"

CONTRACT = [
    ("D1: the release bar",
     "all meet both limits with probability at least 0.5. The owner checks",
     "all meet both limits with probability at least 0.95. The owner checks"),
    ("structural fix (worlds are not stored rows) and no hints: the base rate the ideal observer uses, nothing more",
     "About half of the lots in this program have at least one declared route that meets the requirement, and lots are "
     "selected so that the supplied model's own favourite route passes in only some of them. The supplied models' "
     "probabilities come from their own posterior, in which passing lots are rarer. Use them as hypotheses to test, not "
     "as this lot's odds.",
     "About half of the lots in this program have at least one declared route that meets the requirement. Each lot is "
     "drawn from the posterior the supplied C7.3 package samples, never one of its stored rows, and selected for this "
     "program; in the package's own posterior, passing lots are rarer."),
    ("no hints: the restriction is defined by the sentence before; the coaching after it goes",
     " Your own model rating every route low is not a restriction, because settings you have not tested stay live. If "
     "an affordable observation could still change the conclusion, buy it first.",
     ""),
    ("no hints: irreversibility stated as a fact, not an instruction",
     "Annealing is irreversible: measure a state before you anneal it.",
     "Annealing is irreversible: once a body is annealed, its earlier stage cannot be measured."),
    ("no hints: the physical record's scope, without telling the agent how to use it",
     "They inform your models; they do not bound this lot's routes.",
     "They do not bound this lot's routes."),
    ("no hints: the epistemic coaching goes; the freedom and the authority limits stay",
     "## Standard of evidence\n\nState conclusions at the scope of the states you measured, and mark anything beyond "
     "them as conditional on a model. A route you have not built is untested, whatever a model says about it. Revising "
     "and retaining are both legitimate; say which you did and which observations decided it. No method",
     "## Scope\n\nNo method"),
    ("v1.3 readings",
     "- **Instruments.** Density readings carry the reading sd stated in each return.",
     "- **Instruments.** Density readings report each method's nominal reading sd; this lot's actual scatter is not "
     "stated."),
    ("v1.3 readings: loops and the knee",
     "Loops: independent 2 % log reading errors on Hci and Br, with (BH)max from the same curve,",
     "Loops: independent 2 % log reading errors on Hci and Br; the knee field (Hk) is read with its own error, "
     "including a demagnetization correction fixed per piece, and (BH)max comes from the read curve,"),
    ("v1.3 readings: SEM width and the SPS log",
     "Bodies of one preparation share that preparation's effect; bodies and pieces differ. Their scatter is not stated.",
     "Bodies of one preparation share that preparation's effect; bodies and pieces differ. Their scatter is not stated. "
     "SEM distribution readings (d90, coarse fraction) carry a width read from their own field set. The SPS log's "
     "geometry-derived densities carry a calibration error of about 2 %."),
    ("grading: forecasts are recorded and compared, never graded (Jarrod, Oct 26 and Oct 28)",
     "a forecast without a matching stage, or without a positive sd, is not scored)",
     "a forecast without a matching stage, or without a positive sd, is not compared with its return)"),
    ("grading: forecasts are not part of the grade",
     "Forecasts are scored against the returns.",
     "Each forecast is compared with its return in `inspect_feedback`; forecasts are not part of the grade."),
    ("D5: the reference body is read once",
     "It costs nothing and is not one of your 12 bodies.",
     "It costs nothing and is not one of your 12 bodies. It is not annealed or measured again: a request that names "
     "`REF-B1` is refused."),
]

LABORATORY = [
    ("environment facts: what /proc shows and what the limits are (v1.2 audit, runtime_exploit)",
     "The sandbox has 6 CPUs and 16 GB of memory; a command that exceeds the memory is killed.",
     "The sandbox has 6 CPUs and 16 GB of memory; a command that exceeds the memory is killed. Its `/proc` reports "
     "the host machine's processors and memory, not these limits. A process left running in the background is not "
     "stopped by the 15-minute command limit; it ends with the investigation."),
    ("D10: no network; the shipped library",
     "Bash has network access for literature and user-local software.",
     "Bash has no network access; the literature and records this investigation uses are in `library/` "
     "(`read_sources`, `read_evidence`)."),
]

COVERAGE = [
    ("no hints: drop the list of declared values not yet built (an exploration checklist)",
     "text = economy.coverage_markdown(state)\n",
     "text = economy.coverage_markdown(state)\n"
     "# v1.3 (no hints): the agent's own spend, bodies and looped routes only; the list of declared values not yet built\n"
     "# is an exploration checklist and is left out (env/port_env_text.py).\n"
     "head, cut, tail = text.partition(\"\\n## Declared values not yet built in any body\\n\")\n"
     "if cut:\n"
     "    text = head + \"\\n\" + tail[tail.index(\"This table counts\"):]\n"),
]


def apply(src, edits, name):
    out = src
    for label, old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{name}: edit '{label}' expected 1 match, found {n}")
        out = out.replace(old, new)
    return out


def main():
    diffs = []
    for fname, edits in (("CONTRACT.md", CONTRACT), ("LABORATORY.md", LABORATORY)):
        src = (V12 / fname).read_text()
        out = apply(src, edits, fname)
        (HERE / fname).write_text(out)
        diffs += difflib.unified_diff(src.splitlines(keepends=True), out.splitlines(keepends=True),
                                      f"sdl2-v1.2-20261026/env/{fname}", f"sdl2-v1.3-20261028/env/{fname}")
    src = (V12 / "coverage.py").read_text()
    out = apply(src, COVERAGE, "coverage.py")
    (HERE / "coverage_v13.py").write_text(out)
    diffs += difflib.unified_diff(src.splitlines(keepends=True), out.splitlines(keepends=True),
                                  "sdl2-v1.2-20261026/env/coverage.py", "sdl2-v1.3-20261028/env/coverage_v13.py")
    (HERE / "TEXT_V13.diff").write_text("".join(diffs))
    print(f"wrote CONTRACT.md ({len(CONTRACT)} edits), LABORATORY.md ({len(LABORATORY)} edits), TEXT_V13.diff")


if __name__ == "__main__":
    main()
