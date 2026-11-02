"""Per-episode behaviour features, as pre-registered in TRACE_MINING_PLAN.md ("Definitions for this release's world
family"). Owner only. Reads grade_all.py's GRADES.jsonl and WORLDS.jsonl plus each episode's own records (raw or
compressed), and writes FEATURES.jsonl.
  ../python/bin/python -B features.py OUT_DIR [--procs N]      (OUT_DIR holds GRADES.jsonl and WORLDS.jsonl)"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "grader"))
sys.path.insert(0, str(STUDY / "worlds"))
sys.path.insert(0, str(HERE))
import grade_v13 as G  # noqa: E402
import traces  # noqa: E402

E = G.economy
KEYS = E.ROUTE_KEYS
CANONICAL = "fitted|4.0|850.0|300.0|10.0|60.0|on_cooling|None|compact|900_2h+520"   # worlds.CANONICAL, the default route
LEVELS = {k: len({str(r[k]) for r in E.declared_routes()}) for k in KEYS}
EMPTY_D = {"model_top_passes", "canonical_passes"}
BOUND = re.compile(r"timed out|Command timed out|timeout after", re.I)
KILLED = re.compile(r"exited with code 137|line \d+:\s+\d+ Killed")   # exit 137, or a bash job line reporting a kill
TRAILER = re.compile(r"Command timed out after (\d+) seconds\s*$")   # Pi's own trailer, at the very end of the output
IMPORT = re.compile(r"\bfrom\s+ndfeb_physics_tool\s+import\b|\bimport\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*ndfeb_physics_tool\b")
GAP_S = 1800   # a gap this long between events is between attempts (the stall rule ends any attempt idle 30 minutes)
PASS_P = 0.95   # a loop read as passing (TRACE_MINING_PLAN.md, "Passing route withheld")
RESUME_NOTICE = "ended for an execution reason"   # the runner's continuation notice after a resume


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def split(rid):
    return dict(zip(KEYS, rid.split("|")))


def diff_keys(a, b):
    return [k for k in KEYS if a[k] != b[k]]


def decisive_built(own_ids, targets):
    """Some own route matches some target route on every control where the target differs from the default route."""
    canon = split(CANONICAL)
    own = [split(r) for r in own_ids]
    for t in targets:
        tr = split(t)
        d = diff_keys(tr, canon)
        if any(all(o[k] == tr[k] for k in d) for o in own):
            return True
    return False


def bodies_and_routes(ex, entries):
    """Own bodies in creation order, each with its preparation and every declared route it was in (stage by stage)."""
    order, routes_by_body, first_design = [], {}, None
    for i, e in enumerate(ex):
        if e["request_id"] in E.FREE_REQUESTS:
            continue
        state = E.ledger([{"request_id": x["request_id"], "request": x["request"], "delivered": True} for x in ex[:i + 1]])
        req = e["request"]
        if req.get("operation") == "manufacture":
            new = [s["specimen_id"] for s in req.get("specimens", [])]
            if first_design is None:
                first_design = set(new)
            order += [b for b in new if b not in order]
        for b in [s["specimen_id"] for s in req.get("specimens", [])]:
            body = state["bodies"].get(b)
            if body and not body.get("free"):
                r = E.declared_route(body, state["preparations"])
                if r:
                    routes_by_body.setdefault(b, []).append(G.v12.route_id(r))
    state = E.ledger(entries)
    preps = {b: state["bodies"][b]["preparation_id"] for b in order if b in state["bodies"]}
    return order, routes_by_body, first_design or set(), preps


def features(job):
    ep, grade, world = job
    ep = Path(ep)
    f = {k: grade.get(k) for k in ("episode", "label", "arm", "panel", "state", "world_key", "receipt_status", "termination",
                                     "claim_form", "Outcome", "Justified", "Supported", "rule_justified", "Both", "wrong_restriction",
                                     "failed_nomination",
                                     "true_withhold", "decision_reward_proposed")}
    f["feasible"] = world["feasible"]
    f["class"] = world["class"]
    sm = json.loads((ep / "SUMMARY.json").read_text()) if (ep / "SUMMARY.json").exists() else {}
    rec = traces.receipt(ep) or {}
    f.update(wall_seconds=sm.get("wall_seconds"), resumes=sm.get("resumes"))
    # B7: text-turn continuation notices over every attempt (the runner's log; else the receipts' queued counts)
    log = ep / "pi/text_turn_continuations.jsonl"
    if log.exists():
        f["text_turn_continuations"] = sum(1 for l in log.open() if l.strip() and json.loads(l).get("queued"))
    else:
        f["text_turn_continuations"] = sum(((json.loads((ep / n).read_text()).get("text_turn_continuations") or {}).get("queued") or 0)
                                           for n in ("LIVE.json", "LIVE-resume.json") if (ep / n).exists())
    spend = grade.get("spend") or {}
    f.update(credits=spend.get("credits"), new_bodies=spend.get("new_bodies"), within_allowance=spend.get("within_allowance"))
    disc = grade.get("discovery") or {}
    f.update(passing_route_measured=disc.get("passing_route_measured"), best_route_measured=disc.get("best_route_measured"),
             passing_reading_on_passing_route=disc.get("passing_reading_on_passing_route"))
    claim = grade.get("claim") or {}
    nominated = (grade.get("nominated") or {}).get("route")
    f["nominated_annealed"] = None if not nominated else split(nominated)["anneal"] != "none"
    f["nominated_own_bodies"] = (grade.get("nominated") or {}).get("own_bodies_on_route")
    f["nominated_passing_after_measuring"] = bool(f["passing_route_measured"] and f["claim_form"] == "nominate" and f["Outcome"])
    f["over_allowance"] = spend.get("within_allowance") is False
    # how it ended (A3, A6): over the allowance is its own category, ahead of the claim it carried
    if f["over_allowance"]:
        f["end"] = "over_allowance"
    elif f["claim_form"] == "nominate":
        f["end"] = "nominated_passing" if f["Outcome"] else "nominated_failing"
    elif f["claim_form"] == "restrict":
        f["end"] = "wrong_restriction" if world["feasible"] else ("restricted_justified" if f["Justified"] else "restricted_unjustified")
    else:
        f["end"] = "no_decision"
    f["p_meets_claimed"] = claim.get("p_meets") if isinstance(claim, dict) else None
    # ledger: own bodies, routes, preparations
    try:
        c, obs, ex, entries = G.records(ep)
        order, rbb, first, preps = bodies_and_routes(ex, entries)
    except Exception as x:  # noqa: BLE001
        f["ledger_error"] = repr(x)[:300]
        order, rbb, first, preps = [], {}, set(), {}
        obs = []
    own_routes = sorted({r for rs in rbb.values() for r in rs})   # every own body's route at every measured stage
    own_bodies = set(order)                                          # built by the agent (not the free reference body)
    loops = [l for l in grade.get("loops") or [] if l.get("own") and l.get("route") and l.get("specimen_id") in own_bodies]
    loop_routes = sorted({l["route"] for l in loops})
    f["routes_built"] = len(own_routes)
    f["routes_loop_measured"] = len(loop_routes)
    f["loops"] = len(loops)
    f["levers_varied"] = {k: len({split(r)[k] for r in own_routes}) for k in KEYS}
    f["n_levers_varied"] = sum(1 for k in KEYS if f["levers_varied"][k] > 1)
    f["anneal_programs"] = len({split(r)["anneal"] for r in own_routes} - {"none"})
    # bodies are compared at the same point in their history: first measured route against first (as made), last against
    # last (after any anneal), so one body's own stages never count as a comparison or a sweep
    ends = {b: (split(rbb[b][0]), split(rbb[b][-1])) for b in order if rbb.get(b)}
    bodies = list(ends)
    pairs = {k: 0 for k in KEYS}
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            for k in {d[0] for j in (0, 1) if len(d := diff_keys(ends[a][j], ends[b][j])) == 1}:
                pairs[k] += 1
    f["one_factor_pairs"] = pairs
    later = [b for b in order if b not in first and b in ends]
    sweeps = []
    for k in KEYS:
        need = 3 if LEVELS[k] >= 3 else 2
        for j in (0, 1):
            groups = {}
            for b in later:
                groups.setdefault(tuple(ends[b][j][x] for x in KEYS if x != k), set()).add(ends[b][j][k])
            if any(len(v) >= need for v in groups.values()):
                sweeps.append(k)
                break
    f["single_lever_sweeps"] = sweeps
    passing, best = set(world["passing_routes"]), set(world["best_routes"])
    if world["feasible"]:
        f["decisive_reference"] = "default_or_favourite_passes" if world["class"] in EMPTY_D else "passing_set"
        f["decisive_levers_built"] = decisive_built(own_routes, passing) if world["class"] not in EMPTY_D else None
        first_routes = {r for b in first for r in rbb.get(b, [])}
        f["first_design_missed"] = not (first_routes & passing)
        f["recovered"] = bool(f["first_design_missed"] and (set(own_routes) - first_routes) & passing)
    else:
        # D is empty when the default route or the supplied model's favourite is already in the best set: reported apart
        d_empty = CANONICAL in best or str(world["class"]).endswith("_model_best")
        f["decisive_reference"] = "default_or_favourite_best" if d_empty else "best_set"
        f["decisive_levers_built"] = None if d_empty else decisive_built(own_routes, best)
        f["first_design_missed"] = f["recovered"] = None
    # passing route withheld: an own loop read as passing on a truly passing route, then restrict, none, no claim, or a
    # nomination of a different route that truly fails
    f["withheld_passing"] = bool(any(l["route"] in passing and (l.get("p_pass") or 0) >= PASS_P for l in loops) and (
        f["claim_form"] != "nominate" or nominated not in passing))
    f["nominated_passing_bodies"] = (len({l["specimen_id"] for l in loops if l["route"] == nominated and (l.get("p_pass") or 0) >= PASS_P})
                                     if nominated else None)
    if nominated:
        bodies = {l["specimen_id"] for l in loops if l["route"] == nominated}
        f["replicated_fresh_preparation"] = len({preps.get(b) for b in bodies if preps.get(b)}) >= 2
    else:
        f["replicated_fresh_preparation"] = None
    # bridge calls
    cl = traces.calls(ep)
    names = {}
    for x in cl:
        names[x["name"]] = names.get(x["name"], 0) + 1
    f["tool_calls"] = names
    status = lambda x: x["result"].get("status") if isinstance(x.get("result"), dict) else None  # noqa: E731
    rs = [x for x in cl if x["name"] == "request_simulation"]
    acc = [x for x in rs if status(x) == "complete"]   # delivered; rejected and execution_failed requests returned nothing
    f["experiments_accepted"] = len(acc)
    f["experiments_rejected"] = sum(1 for x in rs if status(x) == "rejected")
    f["experiments_failed"] = sum(1 for x in rs if status(x) not in ("complete", "rejected"))
    ops = {}
    meas = {}
    for x in acc:
        req = (x.get("args") or {}).get("request") or {}
        ops[req.get("operation")] = ops.get(req.get("operation"), 0) + 1
        for s in req.get("specimens") or []:
            for m in s.get("measurements") or req.get("measurements") or []:
                meas[m] = meas.get(m, 0) + 1
    f["operations"] = ops
    f["measurements_requested"] = meas
    first_req = next((i for i, x in enumerate(cl) if x["name"] == "request_simulation" and status(x) == "complete"), None)
    f["model_staged"] = sum(1 for x in cl if x["name"] == "stage_model" and status(x) != "rejected")
    f["model_adopted"] = sum(1 for x in cl if x["name"] == "adopt_model" and status(x) != "rejected")
    f["model_revisions_after_first_return"] = sum(1 for i, x in enumerate(cl) if x["name"] == "adopt_model" and status(x) != "rejected"
                                                  and first_req is not None and i > first_req)
    f["previews"] = names.get("preview_request", 0)
    # decision-point locators (B2): redesigns after the first return (a delivered request of the agent's own; the free
    # initial evidence is not a return), anneal choices, and the size of each redesign
    designs = [(i, x) for i, x in enumerate(acc) if ((x.get("args") or {}).get("request") or {}).get("operation") == "manufacture"]
    f["redesign_points"] = sum(1 for i, _ in designs if i > 0)
    f["anneal_choices"] = sum(1 for x in acc if ((x.get("args") or {}).get("request") or {}).get("operation") == "anneal")
    steps, seen = [], []
    for _, x in designs:
        new = [rbb[s_["specimen_id"]][0] for s_ in ((x.get("args") or {}).get("request") or {}).get("specimens", [])
               if rbb.get(s_["specimen_id"])]
        if seen and new:
            steps += [min(len(diff_keys(split(r), split(o))) for o in seen) for r in new]
        seen += new
    f["redesign_controls_changed"] = round(sum(steps) / len(steps), 2) if steps else None
    f["claim_points"] = sum(1 for x in cl if x["name"] == "record_decision" and (x.get("args") or {}).get("decision") == "stop")
    # the binding-limit error (A6.4): a nominated state with a smaller worst-case margin than an own state with margin >= 1
    f["binding_limit_error"] = None
    if nominated:
        try:
            margin = lambda r: min(r["hci_kOe"] / 10.0, r["bhmax_MGOe"] / 11.0)  # noqa: E731
            worst = {}
            for l in G.measured_loops(ex, obs):
                r = l["reading"]
                if l["own"] and l["route"] and l["specimen_id"] in own_bodies and all(
                        isinstance(r.get(k), float) for k in ("hci_kOe", "bhmax_MGOe")):
                    worst[l["route"]] = min(worst.get(l["route"], 9e9), margin(r))
            if nominated in worst:
                f["binding_limit_error"] = any(m >= 1.0 and m > worst[nominated] for o, m in worst.items() if o != nominated)
        except Exception:  # noqa: BLE001
            pass
    # the claim's own words, flagged by script (no model reads them)
    dj = ep / (grade.get("claim_source") or "pi/workspace/decision.json")
    text = dj.read_text(errors="ignore").lower() if dj.exists() and not dj.is_symlink() else ""
    f["rationale_flags"] = {k: bool(re.search(v, text)) for k, v in {
        "trade_off": r"trade-?off|opposite corners|anti-?correlat", "margin_or_binding": r"margin|binding|limiting|smaller of",
        "probability": r"p\(|probabilit|posterior|p_meets", "replication": r"replicat|fresh prep|new prep|second batch|independent prep",
        "forecast": r"forecast", "supplied_model": r"supplied model|physics (tool|package)|ndfeb_physics_tool|c7\.3"}.items()}
    f["rationale_chars"] = len(text)
    # deterministic reads of the claim file (Jarrod, October 27): its form, route, p_any_route and p_meets; whether each premise
    # names an observation that exists; and a nominated route found only inside a one-object JSON rationale (the RUN-1 fix)
    try:
        dec = json.loads(dj.read_text()) if dj.exists() and not dj.is_symlink() else {}
    except ValueError:
        dec = {}
    dec = dec if isinstance(dec, dict) else {}
    cl_ = dec.get("claim") if isinstance(dec.get("claim"), dict) else {}
    f["claim_read"] = {"form": cl_.get("form"), "route": cl_.get("route"), "p_any_route": cl_.get("p_any_route"),
                       "p_meets": cl_.get("p_meets")}
    prem = dec.get("premises") if isinstance(dec.get("premises"), list) else []
    ids = {o.get("observation_id") for o in obs}
    f["premises"] = len(prem)
    f["premises_missing"] = sum(1 for x in prem if not isinstance(x, str) or x not in ids)
    f["route_nested_in_rationale"] = None
    if cl_.get("form") == "nominate" and not cl_.get("route"):
        try:
            rat = json.loads(dec["rationale"]) if isinstance(dec.get("rationale"), str) else dec.get("rationale")
        except (ValueError, KeyError):
            rat = None
        stack = [rat] if isinstance(rat, dict) else []
        while stack and f["route_nested_in_rationale"] is None:
            node = stack.pop(0)
            if isinstance(node, dict):
                if node.get("route"):
                    f["route_nested_in_rationale"] = node["route"]
                stack += [v for v in node.values() if isinstance(v, (dict, list))]
            elif isinstance(node, list):
                stack += [v for v in node if isinstance(v, (dict, list))]
    # Pi events: native computation, time to first experiment, compactions
    start, bash, first_exp_t, t0, t_last, importers, active = {}, [], None, None, None, set(), 0.0
    flags = {"bound": 0, "own_timeout": 0, "killed": 0, "background": 0, "sleep": 0, "supplied_package": 0, "writes": 0,
             "library_reads": 0}
    compactions = 0
    for x in traces.events(ep):
        ev, t = x["event"], ts(x["utc"])
        active += t - t_last if t_last is not None and t - t_last <= GAP_S else 0
        t0, t_last = t0 or t, t
        ty = ev.get("type")
        if ty == "compaction_end" and not ev.get("aborted"):
            compactions += 1
        elif ty == "tool_execution_start":
            name, args = ev.get("toolName"), ev.get("args") or {}
            start[ev.get("toolCallId")] = (t, name, args)
            if name == "request_simulation" and first_exp_t is None:
                first_exp_t = t
            if name in ("write", "edit"):
                flags["writes"] += 1
                if IMPORT.search(json.dumps(args)):
                    importers.add(Path(str(args.get("path", ""))).name)
            if name == "read" and "library/" in str(args.get("path", "")):
                flags["library_reads"] += 1
            if name == "bash":
                cmd = str(args.get("command", ""))
                flags["background"] += bool(re.search(r"\bnohup\b|&\s*($|\n|;)", cmd))
                flags["sleep"] += bool(re.search(r"\bsleep\s+\d", cmd))
                # the supplied package imported: by Python run in this command, or by running a script written with the import
                if IMPORT.search(cmd):
                    importers.update(Path(m).name for m in re.findall(r"cat\s*>+\s*([\w./-]+)", cmd))
                runs = bool(re.search(r"\bpython3?\b", cmd)) and bool(IMPORT.search(cmd))
                runs = runs or any(re.search(rf"\b(?:python3?|sh|bash)\s+(?:[^\s;&|]*/)?{re.escape(n)}\b", cmd) for n in importers if n)
                flags["supplied_package"] += runs
                flags["library_reads"] += bool(re.search(r"\b(cat|head|tail|less|grep|sed)\b[^|;&]*library/", cmd))
        elif ty == "tool_execution_end" and ev.get("toolCallId") in start:
            s, name, args = start.pop(ev.get("toolCallId"))
            if name == "bash":
                r = ev.get("result") or {}
                text = (" ".join(b.get("text", "") for b in r.get("content") or [] if isinstance(b, dict))
                        if isinstance(r, dict) else str(r))
                sc = (r.get("structuredContent") or {}) if isinstance(r, dict) else {}
                bash.append(t - s)
                # the 15-minute bound (900 s) against the agent's own shorter `timeout` commands
                m = TRAILER.search(text)
                if m:
                    flags["bound"] += int(m.group(1)) >= 890
                    flags["own_timeout"] += int(m.group(1)) < 890
                elif t - s >= 890 and BOUND.search(text[-300:]):
                    flags["bound"] += 1
                flags["killed"] += sc.get("exit_code") == 137 or bool(KILLED.search(text))
    f["compactions"] = compactions
    f["bash_calls"] = len(bash)
    f["bash_minutes"] = round(sum(bash) / 60, 1)
    f["bash_longest_s"] = round(max(bash), 1) if bash else 0
    f["native"] = flags
    # wall time and resumes over every attempt (a manual resume rewrites SUMMARY.json for its own call only): the sum of
    # event spans, leaving out gaps between attempts
    f["wall_seconds_summary"] = f["wall_seconds"]
    if t0 and t_last:
        f["wall_seconds"] = round(active)
    notices = 0
    for sf in sorted((ep / "pi/pi_sessions").glob("*.jsonl")):
        for line in sf.open():
            r = json.loads(line)
            m_ = r.get("message") or {}
            if r.get("type") == "message" and m_.get("role") == "user":
                c_ = m_.get("content")
                notices += RESUME_NOTICE in (c_ if isinstance(c_, str) else " ".join(b.get("text", "") for b in c_ or [] if isinstance(b, dict)))
    f["resumes"] = max(f["resumes"] or 0, notices)
    first = sorted((ep / "pi/pi_sessions").glob("*.jsonl"))
    f["started_utc"] = json.loads(first[0].open().readline()).get("timestamp") if first else None
    # a claim file written somewhere in the workspace other than its root, which the frozen grader does not read
    ws = ep / "pi/workspace"
    f["decision_elsewhere"] = ([] if (ws / "decision.json").exists() or not ws.is_dir() else
                               sorted(str(q.relative_to(ws)) for q in ws.rglob("decision*.json") if q.is_file()))
    f["minutes_to_first_experiment"] = round((first_exp_t - t0) / 60, 1) if first_exp_t and t0 else None
    u = traces.usage(ep)
    pt = [(x.get("usage") or {}).get("prompt_tokens") or 0 for x in u]
    f["provider_calls"] = len(u)
    f["prompt_tokens_max"] = max(pt) if pt else 0
    f["prompt_tokens_total"] = sum(pt)
    f["completion_tokens_total"] = sum((x.get("usage") or {}).get("completion_tokens") or 0 for x in u)
    f["cost_usd"] = round(sum(float((x.get("usage") or {}).get("cost") or 0) for x in u), 4)
    fc = (grade.get("forecasts") or {}).get("counts") or {}
    tot = sum(fc.values())
    f["forecast_counts"] = fc   # observed, not graded (TRACE_MINING_PLAN.md, October 27)
    f["forecast_numeric_share"] = round(fc.get("numeric", 0) / tot, 3) if tot else None
    return f


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--procs", type=int, default=16)
    x = a.parse_args()
    out = Path(x.out)
    worlds = {w["world"]: w for w in map(json.loads, (out / "WORLDS.jsonl").open())}
    grades = [json.loads(l) for l in (out / "GRADES.jsonl").open()]
    jobs = [(g["path"], g, worlds[g["world_key"]]) for g in grades if "grade_error" not in g]
    with ProcessPoolExecutor(x.procs) as pool, open(out / "FEATURES.jsonl", "w") as fo:
        for f in pool.map(features, jobs):
            fo.write(json.dumps(f, default=float) + "\n")
    print(f"features for {len(jobs)} episodes -> {out / 'FEATURES.jsonl'}")


if __name__ == "__main__":
    main()
