"""Leaderboard, primary comparisons, failure anatomy and behaviour comparison from FEATURES.jsonl
(comparisons A1–A7, B1–B7). Owner only. Writes RESULTS.json and REPORT.md next to the features.
  ../python/bin/python -B report.py OUT_DIR [QUEUE.json M]                 the board and the readouts
  ../python/bin/python -B report.py OUT_DIR QUEUE.json M --pair A B        one paired comparison, for a page sentence
  ... --justified JUSTIFIED.jsonl --declared PRIOR [--credits-to-justified C2J.jsonl] [--floor-justified FLOOR_JUSTIFIED.jsonl]
      rung 2, measure 1 (one row per episode: episode, credits_to_justified or null) and the zero-experiment floor
      episodes' rung 2; also writes HEADLINE.md, the headline board
With QUEUE.json and M (the reduced design), the board ranks every arm on
the shared worlds, order 0..M, by Both, Outcome and Justified with exact intervals, and every per-arm readout uses those
worlds. There is no all-pairs table and no Holm correction. A sentence saying one arm beats another is backed by --pair:
exact McNemar and the world bootstrap, on the board's worlds."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

HANDOFF = "2026-10-27T00:52:45Z"   # concurrency lowered from 40 to 20 per host; [episodes, bound kills]
# the 20-worker launchers started 20 episodes at once on each of two hosts
BURSTS = (("2026-10-27T01:51:00Z", "2026-10-27T01:53:00Z"), ("2026-10-27T02:07:00Z", "2026-10-27T02:09:00Z"))
FILL = "2026-10-27T08:46:57Z"   # STOP for the reduced design; episodes started after it are the fill launchers'
HANDOFF2 = "2026-10-27T02:31:53Z"   # lowered from 20 to 12 per host, four 3-worker launchers per host, staggered

ORDER = ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-astra", "google/gemini-4-argon",
         "deepseek/deepseek-v4.1-flash-1017", "bo", "doe"]
SHORT = {"anthropic/claude-opus-5.5": "Opus 5.5", "anthropic/claude-sonnet-5.5": "Sonnet 5.5", "openai/gpt-6-astra": "GPT-6 Astra",
         "google/gemini-4-argon": "Gemini 4 Argon", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek 1017 (base)",
         "bo": "BO", "doe": "DOE"}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [round(float(lo), 4), round(float(hi), 4)]


def rate(xs):
    xs = [bool(x) for x in xs if x is not None]
    return {"k": sum(xs), "n": len(xs), "rate": round(sum(xs) / len(xs), 4) if xs else None, "ci95": cp(sum(xs), len(xs)) if xs else None}


def relabel(F, J=None, declared=None):
    """Justified is Supported (SUPPORTED.jsonl rows, mining/supported.py; the row's standard is its K).
    The board's measures: Outcome from the frozen grader; Justified from rung 2 (JUSTIFIED.jsonl at the
    declared prior; a claim of none, or no claim, is never justified); Both = Outcome and Justified. The grader's own
    scripted Justified is kept as the diagnostic `evidence_rule`. Without rung 2, Justified and Both are None (pending)."""
    for f in F:
        f["evidence_rule"] = f.get("evidence_rule", f.get("rule_justified", f.get("Justified")))   # current field name
        j = None if J is None else (False if f["claim_form"] not in ("nominate", "restrict") else J.get((f["episode"], declared)))
        f["Justified"], f["Both"] = j, (None if j is None else bool(f["Outcome"] and j))
        if f["end"] in ("restricted_justified", "restricted_unjustified", "restricted_pending"):
            f["end"] = "restricted_pending" if j is None else "restricted_justified" if j else "restricted_unjustified"
    return F


def measure(fs):
    """Both once rung 2 is in for every episode, else Outcome (labelled where it is used)."""
    return "Both" if fs and all(f["Both"] is not None for f in fs) else "Outcome"


def paired(by_arm, a, b, reps=10000, seed=20261026, key=None):
    """One measure (Both, or Outcome while rung 2 is pending), paired by world on the worlds both arms ran: difference,
    exact McNemar, world bootstrap."""
    key = key or measure(by_arm.get(a, []) + by_arm.get(b, []))
    wa = {f["world_key"]: bool(f[key]) for f in by_arm.get(a, [])}
    wb = {f["world_key"]: bool(f[key]) for f in by_arm.get(b, [])}
    shared = sorted(set(wa) & set(wb))
    if not shared:
        return None
    x = np.array([wa[w] for w in shared], float)
    y = np.array([wb[w] for w in shared], float)
    n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), (reps, len(shared)))
    boot = (x[idx] - y[idx]).mean(axis=1)
    return {"a": a, "b": b, "measure": key, "worlds": len(shared), "a_rate": round(x.mean(), 4), "b_rate": round(y.mean(), 4),
            "difference": round(float(x.mean() - y.mean()), 4), "a_only": n10, "b_only": n01, "mcnemar_p": round(float(p), 5),
            "bootstrap95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)]}


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(float(np.mean(xs)), 2) if xs else None


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(float(np.median(xs)), 2) if xs else None


def share(fs, key):
    xs = [f.get(key) for f in fs if f.get(key) is not None]
    return round(sum(bool(x) for x in xs) / len(xs), 3) if xs else None


def arm_rows(fs):
    feas = [f for f in fs if f["feasible"]]
    infe = [f for f in fs if not f["feasible"]]
    claims = {}
    for f in fs:
        c = f["claim_form"] if f["claim_form"] in ("nominate", "restrict") else "no_decision"
        claims[c] = claims.get(c, 0) + 1
    # how it ended (A3): a partition of each world type; over the allowance is its own category (features.py, "end")
    end = {k: sum(1 for f in feas if f["end"] == k) for k in
           ("nominated_passing", "nominated_failing", "wrong_restriction", "no_decision", "over_allowance")}
    end_inf = {k: sum(1 for f in infe if f["end"] == k) for k in
               ("restricted_justified", "restricted_unjustified", "restricted_pending", "nominated_failing", "no_decision",
                "over_allowance")}
    dec = [f for f in feas if f.get("decisive_reference") == "passing_set"]
    dec_inf = [f for f in infe if f.get("decisive_reference") == "best_set"]
    noms = [f for f in fs if f["claim_form"] == "nominate"]
    frac = lambda xs, n: {"k": xs, "n": n, "share": round(xs / n, 3) if n else None}  # noqa: E731
    fc = {}
    for f in fs:
        for k, v in (f.get("forecast_counts") or {}).items():
            fc[k] = fc.get(k, 0) + v
    return {
        # no final decision counts as not justified, so Both and Justified are False there (the published convention)
        "n": len(fs), "Both": rate(f["Both"] for f in fs), "Outcome": rate(bool(f["Outcome"]) for f in fs),
        "Justified": rate(f["Justified"] for f in fs), "evidence_rule": rate(bool(f["evidence_rule"]) for f in fs),
        "by_lot": {k: {"n": len(g), "Outcome": rate(bool(f["Outcome"]) for f in g), "Justified": rate(f["Justified"] for f in g),
                       "Both": rate(f["Both"] for f in g)} for k, g in (("feasible", feas), ("infeasible", infe))},
        "claims": claims,
        "feasible_worlds": {"n": len(feas), "how_it_ended": end,
                            "passing_route_measured": sum(1 for f in feas if f["passing_route_measured"]),
                            "nominated_passing_after_measuring": sum(1 for f in feas if f["nominated_passing_after_measuring"]),
                            "withheld_passing": sum(1 for f in feas if f["withheld_passing"]),
                            "decisive_levers_built": f"{sum(1 for f in dec if f['decisive_levers_built'])}/{len(dec)}",
                            "default_or_favourite_passes_worlds": sum(1 for f in feas if f.get("decisive_reference") == "default_or_favourite_passes"),
                            "first_design_missed": sum(1 for f in feas if f["first_design_missed"]),
                            "recovered": sum(1 for f in feas if f["recovered"]),
                            "annealed_nomination_among_both": f"{sum(1 for f in feas if f['Both'] and f['nominated_annealed'])}/"
                                                              f"{sum(1 for f in feas if f['Both'] and f['claim_form'] == 'nominate')}"},
        "infeasible_worlds": {"n": len(infe), "how_it_ended": end_inf,
                              "best_route_measured": sum(1 for f in infe if f["best_route_measured"]),
                              "decisive_levers_built": f"{sum(1 for f in dec_inf if f['decisive_levers_built'])}/{len(dec_inf)}",
                              "default_or_favourite_best_worlds": sum(1 for f in infe if f.get("decisive_reference") == "default_or_favourite_best")},
        # failure anatomy (A6), as shares of the episodes each item can apply to
        "failure_anatomy": {
            "decisive_levers_never_built": frac(sum(1 for f in dec if f["decisive_levers_built"] is False), len(dec)),
            "passing_route_withheld": frac(sum(1 for f in feas if f["withheld_passing"]), len(feas)),
            "nomination_without_replication": frac(sum(1 for f in noms if (f.get("nominated_passing_bodies") or 0) < 2), len(noms)),
            "binding_limit_misjudged": frac(sum(1 for f in fs if f.get("binding_limit_error") is True),
                                            sum(1 for f in fs if f.get("binding_limit_error") is not None)),
            "wrong_restriction": frac(sum(1 for f in feas if f["wrong_restriction"]), len(feas))},   # the claim, whatever the spend
        # decision-point frequency (B2): mean occurrences per episode
        "decision_points": {"redesign_points": mean(f["redesign_points"] for f in fs), "anneal_choices": mean(f["anneal_choices"] for f in fs),
                            "claim_points": mean(f["claim_points"] for f in fs),
                            "redesign_controls_changed": median(f["redesign_controls_changed"] for f in fs)},
        # compaction (B6) and text-only ends and continuations (B7)
        "compaction": {k: {"n": len(g), "Outcome": rate(bool(f["Outcome"]) for f in g), "Both": rate(f["Both"] for f in g),
                           "claims": {c: sum(1 for f in g if (f["claim_form"] if f["claim_form"] in ("nominate", "restrict") else "no_decision") == c)
                                      for c in ("nominate", "restrict", "no_decision")}}
                       for k, g in (("compacted", [f for f in fs if f["compactions"]]), ("not_compacted", [f for f in fs if not f["compactions"]]))},
        "text_only_ends": {"ended_without_a_claim": frac(sum(1 for f in fs if f["claim_form"] not in ("nominate", "restrict", "none")), len(fs)),
                           "used_continuation_notice": frac(sum(1 for f in fs if f.get("text_turn_continuations")), len(fs))},
        "over_allowance": sum(1 for f in fs if f["over_allowance"]),
        "behaviour": {
            "routes_built": median(f["routes_built"] for f in fs), "loops": median(f["loops"] for f in fs),
            "levers_varied": median(f["n_levers_varied"] for f in fs),
            "one_factor_pairs": median(sum(f["one_factor_pairs"].values()) for f in fs),
            "single_lever_sweep_share": round(sum(1 for f in fs if f["single_lever_sweeps"]) / len(fs), 3) if fs else None,
            "anneal_programs": median(f["anneal_programs"] for f in fs),
            "replicated_nomination_share": share([f for f in fs if f["claim_form"] == "nominate"], "replicated_fresh_preparation"),
            "experiments_accepted": median(f["experiments_accepted"] for f in fs),
            "experiments_rejected": mean(f["experiments_rejected"] for f in fs),
            "experiments_failed": mean(f.get("experiments_failed") for f in fs),
            "credits": median(f["credits"] for f in fs), "bodies": median(f["new_bodies"] for f in fs),
            "minutes_to_first_experiment": median(f["minutes_to_first_experiment"] for f in fs),
            "models_adopted": median(f["model_adopted"] for f in fs),
            "model_revisions_after_first_return": median(f["model_revisions_after_first_return"] for f in fs),
            "previews": median(f["previews"] for f in fs),
            # observed, not graded: the pooled share of returned quantities with a numeric, stage-matched forecast
            "forecast_numeric_share": round(fc.get("numeric", 0) / sum(fc.values()), 3) if sum(fc.values()) else None,
            "bash_minutes": median(f["bash_minutes"] for f in fs),
            "background_job_share": round(sum(1 for f in fs if f["native"]["background"]) / len(fs), 3) if fs else None,
            "supplied_package_use_share": round(sum(1 for f in fs if f["native"]["supplied_package"]) / len(fs), 3) if fs else None,
            "commands_stopped_by_bound": sum(f["native"]["bound"] for f in fs),
            "commands_stopped_by_own_timeout": sum(f["native"].get("own_timeout", 0) for f in fs),
            "claim_file_outside_workspace_root": sum(1 for f in fs if f.get("decision_elsewhere")),
            "commands_killed_by_memory": sum(f["native"]["killed"] for f in fs),
            "compaction_share": round(sum(1 for f in fs if f["compactions"]) / len(fs), 3) if fs else None},
        "cost": {"usd_total": round(sum(f["cost_usd"] or 0 for f in fs), 2), "usd_median": median(f["cost_usd"] for f in fs),
                 "provider_calls_median": median(f["provider_calls"] for f in fs),
                 "prompt_tokens_max_median": median(f["prompt_tokens_max"] for f in fs),
                 "wall_minutes_median": median((f["wall_seconds"] or 0) / 60 for f in fs if f["wall_seconds"]),
                 "resumes": sum(f["resumes"] or 0 for f in fs)},
    }


def moves_vs_outcome(fs):
    """B3: Both when a move is made against when it is not (descriptive, not causal)."""
    # (move, feasible worlds only): a move that cannot be made in an infeasible world is compared on feasible worlds
    moves = {"single_lever_sweep": (lambda f: bool(f["single_lever_sweeps"]), False),
             "decisive_levers_built": (lambda f: f["decisive_levers_built"], True),
             "passing_route_measured": (lambda f: f["passing_route_measured"], True),
             "replicated_nomination": (lambda f: f["replicated_fresh_preparation"], False),
             "model_revised_after_return": (lambda f: f["model_revisions_after_first_return"] > 0, False),
             "background_jobs": (lambda f: f["native"]["background"] > 0, False)}
    out, key = {"measure": measure(fs)}, measure(fs)
    for name, (fn, feasible_only) in moves.items():
        pool = [f for f in fs if f["feasible"] or not feasible_only]
        made = [bool(f[key]) for f in pool if fn(f) is True]
        not_made = [bool(f[key]) for f in pool if fn(f) is False]
        out[name] = {"made": rate(made), "not_made": rate(not_made)}
    return out


def main():
    if "--p95" in sys.argv:
        return final_main()
    out = Path(sys.argv[1])
    F = [json.loads(l) for l in (out / "FEATURES.jsonl").open()]
    # rung 2: --justified PATH --declared PRIOR (JUSTIFIED.jsonl rows: episode, prior, justified)
    J, declared, priors = None, None, []
    if "--justified" in sys.argv:
        rows = [json.loads(l) for l in Path(sys.argv[sys.argv.index("--justified") + 1]).open() if l.strip()]
        J = {(r["episode"], r.get("standard", r.get("prior"))): bool(r["justified"]) for r in rows if r.get("justified") is not None}
        declared = sys.argv[sys.argv.index("--declared") + 1]
        priors = sorted({r.get("standard", r.get("prior")) for r in rows})
    sensitivity = {}
    for pr in priors:   # Justified and Both under each prior, the declared one included
        G2 = relabel([dict(f) for f in F], J, pr)
        sensitivity[pr] = G2
    relabel(F, J, declared)
    by_arm = {}
    for f in F:
        by_arm.setdefault(f["label"], []).append(f)
    labels = [a for a in ORDER if a in by_arm] + sorted(a for a in by_arm if a not in ORDER)
    board = by_arm
    flags = {"--justified", "--declared", "--pair", "--credits-to-justified", "--floor-justified"}
    pos = [x for i, x in enumerate(sys.argv) if i > 0 and x not in flags and sys.argv[i - 1] not in flags
           and not (i > 1 and sys.argv[i - 2] == "--pair")]
    if len(pos) >= 3:   # the reduced design: the board and per-arm readouts on the shared worlds 0..M
        order = {e["name"]: e["order"] for e in json.loads(Path(pos[1]).read_text())["entries"]}
        m = int(pos[2])
        board = {a: [f for f in fs if order.get(f["episode"], 10**9) <= m] for a, fs in by_arm.items()}
    if "--pair" in sys.argv:   # one paired comparison on the board's worlds, for a sentence that says one arm beats another
        a, b = sys.argv[sys.argv.index("--pair") + 1:sys.argv.index("--pair") + 3]
        p = paired(board, a, b)
        (out / f"PAIR_{a.split('/')[-1]}__{b.split('/')[-1]}.json").write_text(json.dumps(p, indent=1) + "\n")
        print(json.dumps(p, indent=1))
        return
    res = {"board_worlds": sorted({len({f["world_key"] for f in fs}) for fs in board.values()}),
           "justified": {"source": "Supported" if J else "pending", "standard": declared},
           "arms": {a: arm_rows(board[a]) for a in labels}}
    names = {f["episode"] for fs in board.values() for f in fs}
    res["supported_K_sensitivity"] = {
        pr: {a: {"Justified": rate(g["Justified"] for g in G2 if g["label"] == a and g["episode"] in names),
                 "Both": rate(g["Both"] for g in G2 if g["label"] == a and g["episode"] in names)} for a in labels}
        for pr, G2 in sensitivity.items()}
    # record checks (reported, not enforced): grade errors, more than one episode for an arm and world
    G = [json.loads(l) for l in (out / "GRADES.jsonl").open()] if (out / "GRADES.jsonl").exists() else []
    per = {}
    for g in G:
        per[(g.get("label"), g.get("world_key"))] = per.get((g.get("label"), g.get("world_key")), 0) + 1
    res["checks"] = {"grade_rows": len(G), "grade_errors": [g["episode"] for g in G if "grade_error" in g],
                     "arm_world_duplicates": [list(k) for k, n in per.items() if n > 1], "features_rows": len(F),
                     "claims_read_elsewhere": [g["episode"] for g in G if g.get("claim_source")]}
    res["moves_vs_outcome"] = {a: moves_vs_outcome(board[a]) for a in labels}
    base = {f["world_key"]: f for f in by_arm.get("deepseek/deepseek-v4.1-flash-1017", [])}
    head = {}
    for w, f in base.items():
        k = measure(F)
        if not f[k] and any(g["world_key"] == w and g[k] for a in labels if a != f["label"] for g in by_arm[a]):
            head[f["class"]] = head.get(f["class"], 0) + 1
    res["headroom_worlds_by_class"] = head
    # contention covariate: episodes started at 40 per host, in a relaunch burst, or staggered at 20 per host
    res["contention"] = {}
    for a in labels:
        for f in by_arm[a]:
            if f.get("started_utc"):
                t = f["started_utc"]
                k = ("at_40" if t < HANDOFF else "relaunch_burst" if any(x <= t < y for x, y in BURSTS)
                     else "at_20" if t < HANDOFF2 else "at_12" if t < FILL else "fill")
                c = res["contention"].setdefault(a, {"at_40": [0, 0], "relaunch_burst": [0, 0], "at_20": [0, 0], "at_12": [0, 0],
                                                     "fill": [0, 0]})[k]
                c[0] += 1
                c[1] += f["native"]["bound"]
    res["headline"] = headline(out, board, labels, J, declared, order if len(pos) >= 3 else None, m if len(pos) >= 3 else None)
    (out / "RESULTS.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    ci = lambda r: f"{r['k']}/{r['n']} = {r['rate']} [{r['ci95'][0]}, {r['ci95'][1]}]" if r["n"] else "—"  # noqa: E731
    # Rung 1 is led by the material outcome: Outcome on feasible worlds (a nominated exact route that meets the
    # requirement, within the allowance). Correct no-go on infeasible worlds is saturated and not listed;
    # all-worlds Outcome stays to the right because Both is defined over all worlds.
    mat = lambda a: res["arms"][a]["by_lot"]["feasible"]["Outcome"]  # noqa: E731
    ranked = sorted(labels, key=lambda a: -((res["arms"][a]["Both"] if J else mat(a))["rate"] or 0))
    L = ["# SDL-2 sealed leaderboard: mined results", "", f"Board worlds per arm: {res['board_worlds']}.", "",
         f"Justified: {'Supported at ' + str(declared) if J else 'pending; ranked by material outcome'}. "
         "Material outcome: the nominated exact route meets the requirement in the world, within the allowance, on the "
         "feasible worlds. Evidence rule: the v1.2 scripted rule, a diagnostic, never Justified.", "",
         "| Rank | Agent | Both (95% CI) | Material outcome, feasible worlds (95% CI) | Justified (95% CI) | Outcome, all worlds (95% CI) | Evidence rule | Nominate / restrict / none | Wrong restrictions | Median cost |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(ranked, 1):
        r, c = res["arms"][a], res["arms"][a]["claims"]
        L.append(f"| {i} | {SHORT.get(a, a)} | {ci(r['Both'])} | {ci(mat(a))} | {ci(r['Justified'])} | {ci(r['Outcome'])} | {r['evidence_rule']['rate']} | "
                 f"{c.get('nominate', 0)} / {c.get('restrict', 0)} / {c.get('no_decision', 0)} | "
                 f"{r['failure_anatomy']['wrong_restriction']['k']}/{r['feasible_worlds']['n']} | ${r['cost']['usd_median']} |")
    L += ["", "## Behaviour (medians unless a share)", "", "| Behaviour | " + " | ".join(SHORT.get(a, a) for a in labels) + " |",
          "|---|" + "---|" * len(labels)]
    for k in res["arms"][labels[0]]["behaviour"]:
        L.append(f"| {k} | " + " | ".join(str(res["arms"][a]["behaviour"][k]) for a in labels) + " |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "HEADLINE.md").write_text(headline_md(res["headline"], ci) + "\n")
    print("\n".join(L[:12]))


# world classes in order of distance from the supplied model's favourite route (worlds/worlds.py strata), with plain labels
CLASS_LABEL = {"model_top_passes": "Twin's favourite route passes", "canonical_passes": "Default route passes",
               "feedstock": "One lever: feedstock", "temperature_or_hold": "One lever: temperature or hold",
               "rate_or_pressure": "One lever: heating rate or pressure", "anneal": "One lever: anneal",
               "multi": "Several levers", "near_miss_model_best": "No route; near miss, favourite closest",
               "near_miss_best_elsewhere": "No route; near miss, best route elsewhere",
               "clear_model_best": "No route; clear, favourite closest", "clear_best_elsewhere": "No route; clear, best route elsewhere"}


def jsonl(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def headline(out, board, labels, J, declared, order, m):
    """The headline board: Both (the ranking), material outcome on the feasible
    worlds, and credits to a justified decision (measure 1); two zero-experiment floor rows; Both by
    world class. Everything else stays in REPORT.md, the detail table. Inputs that are not in yet are reported as pending."""
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
    c2j = {r["episode"]: r.get("credits_to_supported", r.get("credits_to_justified")) for r in jsonl(arg("--credits-to-justified"))} if arg("--credits-to-justified") else None
    fj = ({(r["episode"], r.get("standard", r.get("prior"))): bool(r["justified"]) for r in jsonl(arg("--floor-justified")) if r.get("justified") is not None}
          if arg("--floor-justified") else None)
    key = "Both" if J else "Outcome"
    arms = {}
    for a in labels:
        fs = board[a]
        mat = rate(bool(f["Outcome"]) for f in fs if f["feasible"])
        both_ok = [f for f in fs if f["Both"]]
        vals = sorted(c2j[f["episode"]] for f in both_ok if c2j and c2j.get(f["episode"]) is not None)
        arms[a] = {"Both": rate(f["Both"] for f in fs) if J else None, "material_outcome": mat,
                   "credits_to_justified": None if c2j is None else
                   {"median": vals[len(vals) // 2] if len(vals) % 2 else (sum(vals[len(vals) // 2 - 1:len(vals) // 2 + 1]) / 2 if vals else None),
                    "n": len(vals), "of_both_correct": len(both_ok)}}
    floor = None
    if order is not None and (out / "FLOOR.json").exists():
        rows = [r for r in json.loads((out / "FLOOR.json").read_text())["rows"] if r["order"] <= m]
        floor = {}
        for policy, ok in (("restrict", "restrict_all_outcome"), ("top", "top_route_outcome")):
            js = [None if fj is None else fj.get((f"floor-{policy}-o{r['order']:02d}", declared)) for r in rows]
            both = None if any(j is None for j in js) else rate(bool(r[ok]) and j for r, j in zip(rows, js))
            floor[policy] = {"Both": both, "material_outcome": rate(bool(r[ok]) for r in rows if r["feasible"]),
                             "worlds": len(rows)}
    seen = {f["class"] for a in labels for f in board[a]}
    classes = [c for c in CLASS_LABEL if c in seen] + sorted(seen - set(CLASS_LABEL), key=str)
    by_class = {c: {a: rate(f[key] for f in board[a] if f["class"] == c) for a in labels} for c in classes}
    ranked = sorted(labels, key=lambda a: -(((arms[a]["Both"] if J else arms[a]["material_outcome"]) or {}).get("rate") or 0))
    return {"ranked_by": "Both" if J else "material outcome (Supported pending)", "declared_prior": declared, "order": ranked,
            "arms": arms, "floor": floor, "by_class_measure": key, "by_class": by_class}


def headline_md(h, ci):
    pct = lambda r: "pending" if r is None or not r["n"] else f"{100 * r['k'] / r['n']:.0f}% ({r['k']}/{r['n']})"  # noqa: E731
    cj = lambda c: ("pending" if c is None else "—" if c["median"] is None else  # noqa: E731
                    f"{c['median']:.0f} (median of {c['n']})")
    L = ["# SDL-2 leaderboard", "",
         f"Ranked by {h['ranked_by']}. **Both:** the final claim is right in the world and the agent's own evidence "
         "supports it (a restriction measured routes among the world's 100 nearest to passing and every such loop failed; a "
         "nomination passed on two bodies of the route). **Material outcome:** on the feasible worlds, the agent nominated an exact "
         "process that meets 10 kOe and 11 MGOe. **Credits to a supported decision:** among its Both-correct episodes, the "
         "credits spent until its final claim was supported (114-credit steps; allowance 570). Exact 95% intervals in RESULTS.json.", "",
         "| Rank | Agent | Both | Material outcome | Credits to a supported decision |", "|---|---|---|---|---|"]
    for i, a in enumerate(h["order"], 1):
        x = h["arms"][a]
        L.append(f"| {i} | {SHORT.get(a, a)} | {pct(x['Both'])} | {pct(x['material_outcome'])} | {cj(x['credits_to_justified'])} |")
    if h["floor"]:
        for policy, label in (("restrict", "No experiments: restrict everything"),
                              ("top", "No experiments: the twin's favourite route, untested")):
            x = h["floor"][policy]
            L.append(f"| — | *{label}* | {pct(x['Both'])} | {pct(x['material_outcome'])} | 0 (no experiments) |")
    L += ["", f"## {h['by_class_measure']} by world class (how far the lot is from the twin's favourite route)", "",
          "| World class | " + " | ".join(SHORT.get(a, a) for a in h["order"]) + " |", "|---|" + "---|" * len(h["order"])]
    for c, row in h["by_class"].items():
        L.append(f"| {CLASS_LABEL.get(c, c)} | " + " | ".join(f"{row[a]['k']}/{row[a]['n']}" for a in h["order"]) + " |")
    L += ["", "Detail (all-worlds Outcome, Justified, evidence rule, claims, wrong restrictions, cost, behaviour): REPORT.md."]
    return "\n".join(L)


def complete_rate(values):
    """Keep the full denominator; a missing verdict never disappears from a rate."""
    values = list(values)
    missing = sum(v is None for v in values)
    if missing:
        return {"k": sum(v is True for v in values), "n": len(values),
                "rate": None, "ci95": None, "unresolved": missing}
    return {**rate(values), "unresolved": 0}


def final_board(grades, floors, posterior_rows, prior="declared"):
    """Definition B board. Inputs are derived truth grades and the separate frozen posterior."""
    grouped = {}
    for r in posterior_rows:
        grouped.setdefault((r["episode"], r["prior"]), []).append(r)
    posterior = {key: rs[0] for key, rs in grouped.items() if len(rs) == 1}
    all_grades = [dict(r) for r in grades]
    unresolved = []
    for r in all_grades:
        j = posterior.get((r["episode"], prior), {})
        has_claim = r["claim_form"] in ("nominate", "restrict")
        r["Justified"] = j.get("justified") if has_claim else False
        r["Justified_contract"] = j.get("justified_contract") if has_claim else False
        r["Both"] = None if r["Justified"] is None else bool(r["Outcome"] and r["Justified"])
        r["Both_contract"] = None if r["Justified_contract"] is None else bool(r["Outcome_contract"] and r["Justified_contract"])
        if r["Justified"] is None or not j:
            unresolved.append(r["episode"])
    shared = [r for r in all_grades if r["order"] <= 51]
    labels = [lab for lab in ORDER if any(r["label"] == lab for r in shared)]
    labels += sorted({r["label"] for r in shared} - set(labels))
    arms = {}
    for lab in labels:
        rs = [r for r in shared if r["label"] == lab]
        both = [r for r in rs if r["Both"] is True]
        pending = any(r["Both"] is None for r in rs)
        arms[lab] = {
            "Both": complete_rate(r["Both"] for r in rs),
            "material_outcome": complete_rate(r["Outcome"] and r["claim_form"] == "nominate" for r in rs if r["material_feasible"]),
            "credits": {"median": None if pending else median(r["spend"]["credits"] for r in both),
                        "n": len(both), "pending": pending},
            "Outcome": complete_rate(r["Outcome"] for r in rs),
            "Justified": complete_rate(r["Justified"] for r in rs),
            "contract": {"Both": complete_rate(r["Both_contract"] for r in rs),
                         "material_outcome": complete_rate(r["Outcome_contract"] and r["claim_form"] == "nominate" for r in rs if r["contract_feasible"]),
                         "Outcome": complete_rate(r["Outcome_contract"] for r in rs),
                         "Justified": complete_rate(r["Justified_contract"] for r in rs)},
        }
    floor_rows = [r for r in floors if r["order"] <= 51]
    floor_metrics = {}
    for policy, outcome in [("restrict", "restrict_all_outcome"), ("top", "top_route_outcome")]:
        js = [posterior.get((f"floor-{policy}-o{r['order']:02d}", prior), {}).get("justified") for r in floor_rows]
        for r, j in zip(floor_rows, js):
            if j is None:
                unresolved.append(f"floor-{policy}-o{r['order']:02d}")
        floor_metrics[policy] = {
            "Both": complete_rate(None if j is None else bool(r[outcome] and j) for r, j in zip(floor_rows, js)),
            "material_outcome": complete_rate(policy == "top" and r[outcome] for r in floor_rows if r["feasible"]),
            "credits": {"median": 0, "n": len(floor_rows), "pending": False},
        }
    classes = [c for c in CLASS_LABEL if any(r["class"] == c for r in shared)]
    classes += sorted({r["class"] for r in shared} - set(classes))
    by_class = {c: {lab: complete_rate(r["Both"] for r in shared if r["class"] == c and r["label"] == lab)
                    for lab in labels} for c in classes}
    complete = not unresolved
    # Pending rows have no ranking. Stable tie order is not a secondary metric.
    ranked = sorted(labels, key=lambda lab: -arms[lab]["Both"]["k"]) if complete else labels
    straddles = [{"episode": r["episode"], "label": r["label"], "order": r["order"],
                  **r["nomination"]} for r in grades if (r.get("nomination") or {}).get("straddles")]
    return {"status": "complete" if complete else "pending rung 2", "ranked_by": "Both" if complete else None,
            "declared_prior": prior, "order": ranked, "arms": arms, "floor": floor_metrics,
            "by_class": by_class, "unresolved": unresolved, "truth_straddles": straddles}


def final_markdown(board):
    def fmt(r):
        if r["unresolved"]:
            return f"pending ({r['unresolved']}/{r['n']} unresolved)"
        return f"{r['k']}/{r['n']}"

    def credit(c):
        return "pending" if c["pending"] else "—" if c["median"] is None else f"{c['median']:g}"

    ranked = board["status"] == "complete"
    lines = ["# SDL-2 v1.2 leaderboard", "",
             "Ranked by **Both**: the final claim is correct in its world and justified by the agent’s own evidence under the declared prior."
             if ranked else "**Not final: rung 2 is incomplete. No ranks are assigned.**", "",
             "| Arm | Both | Material outcome | Credits |", "|---|---:|---:|---:|"]
    for lab in board["order"]:
        r = board["arms"][lab]
        lines.append(f"| {SHORT.get(lab, lab)} | {fmt(r['Both'])} | {fmt(r['material_outcome'])} | {credit(r['credits'])} |")
    for policy, label in [("restrict", "No experiments: restrict everything"), ("top", "No experiments: supplied model’s top route")]:
        r = board["floor"][policy]
        lines.append(f"| *{label}* | {fmt(r['Both'])} | {fmt(r['material_outcome'])} | 0 |")
    lines += ["", "Material outcome counts nominations with P(all three bodies from a fresh preparation meet both material limits) ≥ 0.95, within the allowance, among worlds where such a route exists. Restrictions remain judged against the task’s original 0.5 bar. Credits are median final credits spent on Both-correct claims; they do not measure when justification was first reached. Allowance: 570 credits and 12 bodies. The zero-experiment floors retain the same free reference measurement.",
              "", "**Disclosures.** The 0.95 bar was applied after the runs; agents were told 0.5. The original 96-world panel was designed to be half feasible at the contract bar; its mix was fixed late. The 52 shared leaderboard worlds contain 25 contract-feasible worlds. This measures decisions in the declared simulated world family, not physical process qualification. Ordering is descriptive.",
              "", "## Both by world class", "", "Classes retain their original contract-bar definitions.", "",
              "| World class | " + " | ".join(SHORT.get(lab, lab) for lab in board["order"]) + " |",
              "|---|" + "---:|" * len(board["order"])]
    for c, row in board["by_class"].items():
        lines.append("| " + CLASS_LABEL.get(c, c) + " | " + " | ".join(fmt(row[lab]) for lab in board["order"]) + " |")
    lines += ["", "Exact binomial intervals and underlying counts: RESULTS.json. The original 0.5 board, unresolved rows, and Monte Carlo limits: REPORT.md."]
    return "\n".join(lines) + "\n"


def final_main():
    """report.py OUT --p95 GRADES_P95.jsonl --floor-p95 FLOOR_P95.json --justified JUSTIFIED.jsonl"""
    import argparse
    import hashlib
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--p95", required=True)
    p.add_argument("--floor-p95", required=True)
    p.add_argument("--justified")
    p.add_argument("--declared", default="declared")
    p.add_argument("--measure2")
    a = p.parse_args()
    grades = jsonl(a.p95)
    floors = json.loads(Path(a.floor_p95).read_text())["rows"]
    rows = jsonl(a.justified) if a.justified else []
    b = final_board(grades, floors, rows, a.declared)
    counts = {}
    for r in rows:
        key = (r["episode"], r["prior"])
        counts[key] = counts.get(key, 0) + 1
    expected = {r["episode"] for r in grades} | {f"floor-{policy}-o{i:02d}" for policy in ("restrict", "top") for i in range(52)}
    declared_ids = {r["episode"] for r in rows if r["prior"] == a.declared}
    all_unresolved = [r["episode"] for r in rows if r["prior"] == a.declared and r.get("justified") is None]
    b["input_audit"] = {"expected_unique_episodes": len(expected), "observed_unique_episodes": len(declared_ids),
                        "missing": sorted(expected - declared_ids), "unexpected": sorted(declared_ids - expected),
                        "duplicates": [list(k) for k, n in counts.items() if n > 1],
                        "no_verdict": sorted(set(all_unresolved)), "posterior_rows": len(rows)}
    b["provenance"] = {"inputs": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                   for path in [a.p95, a.floor_p95, a.justified, a.measure2] if path},
                       "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "HEADLINE.md").write_text(final_markdown(b))
    details = ["# SDL-2 v1.2 board details", "", "The headline uses definition B: nomination reliability ≥ 0.95, restriction truth at the original 0.5 contract bar. The prior is unchanged. Both requires the separate calibrated rung-2 posterior; the frozen grader’s scripted evidence rule is not a substitute.",
               "", "## Original 0.5 contract board", "", "| Arm | Both | Material outcome | Outcome, all worlds | Justified |", "|---|---:|---:|---:|---:|"]
    def display(r):
        return "pending" if r["unresolved"] else f"{r['k']}/{r['n']}"
    contract_order = sorted(b["order"], key=lambda lab: -b["arms"][lab]["contract"]["Both"]["k"])
    for lab in contract_order:
        r = b["arms"][lab]["contract"]
        details.append("| " + SHORT.get(lab, lab) + " | " + " | ".join(display(r[k]) for k in ["Both", "material_outcome", "Outcome", "Justified"]) + " |")
    details += ["", "The 0.5 comparison requires its own posterior functional (`justified_contract`); a 0.95 failure is not treated as a 0.5 failure. The original rung-1 grade is reused without recomputation.",
                "", "## Final 0.95 all-world measures", "", "| Arm | Outcome | Justified |", "|---|---:|---:|"]
    for lab in b["order"]:
        r = b["arms"][lab]
        details.append(f"| {SHORT.get(lab, lab)} | {display(r['Outcome'])} | {display(r['Justified'])} |")
    details += ["", "## Truth Monte Carlo limits", "", "Nomination estimates start at 256 owner lots and extend in batches of 16 while the 95% Wilson interval includes 0.95, up to 2,048 lots. At the cap the point estimate decides, as documented before this run. Repeated Wilson checks are not an anytime-valid confidence sequence. World feasibility reuses the accepted counterfactual; supplementary worlds use the original 0.5 passing-route set. No physical measurement is implied.",
                "", "| Episode | Arm | World order | Passing lots / lots | P(all three) | Wilson interval | Nomination passes |", "|---|---|---:|---:|---:|---|---|"]
    for r in b["truth_straddles"]:
        details.append(f"| {r['episode']} | {SHORT.get(r['label'], r['label'])} | {r['order']} | {r['k']}/{r['n']} | {r['p_all']:.6f} | [{r['wilson95'][0]:.6f}, {r['wilson95'][1]:.6f}] | {r['meets']} |")
    details += ["", "The three remaining truth straddles are point-estimate passes: two Sonnet nominations and one DeepSeek nomination. If all fall below 0.95, their material outcomes become 10/17 and 4/17 respectively, rather than 12/17 and 5/17. They affect Both only when the corresponding claims are Justified. The headline retains the predeclared cap rule.", "", "## Posterior completeness", "", "```json", json.dumps(b["input_audit"], indent=2), "```",
                "", "## Stated probability against the posterior", ""]
    if a.measure2:
        details += [Path(a.measure2).read_text()]
    else:
        details += ["Pending the frozen posterior. Agents stated probabilities for the original 0.5 contract; compare them with the matching 0.5 posterior, not the 0.95 functional. BO and DOE values are their surrogate outputs. This diagnostic costs no additional sampling."]
    details += ["", "## What the board establishes", "", "The board compares seven untrained baseline arms on 52 shared simulated worlds: 25 meet the original 0.5 bar and 17 meet the final 0.95 bar. The original 96-world panel has 48 contract-feasible worlds; the reduced 400-episode design covers 64 distinct worlds, of which 34 are contract-feasible. It measures final material decisions and whether the agent’s own experimental record supports them under the declared prior. It does not establish training gains, explanations of behavior, transfer to new world families, or physical process qualification. No claim that one arm beats another is made without the specified paired analysis. The panel composition and the post-run reliability threshold limit interpretation.",
                "", "Credits are final spend among Both-correct claims. Prefix posteriors remain deferred. No new trace mining or qualitative interpretation is included."]
    (out / "REPORT.md").write_text("\n".join(details) + "\n")
    (out / "RESULTS.json").write_text(json.dumps(b, indent=2) + "\n")
    print(json.dumps({"status": b["status"], "arms": len(b["arms"]), "unresolved": len(b["unresolved"]),
                      "truth_straddles": len(b["truth_straddles"])}))


if __name__ == "__main__":
    main()
