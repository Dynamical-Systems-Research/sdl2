"""SDL-2 v1.3 rung-1 grader (built from ../sdl2-v1.2-20261026/grader/grade_v12.py by grader/port_grade.py; diff in
grader/grade_v13.diff). Owner only; reads the episode's authoritative ledgers, never agent-writable files except
decision.json. CLI: python -B grade_v13.py EPISODE_DIR [--procs N]
v1.3 (MEASUREMENT_SPEC.md section 4; grader/RUNG2_DECISION.md, option B, adopted Oct 29): Outcome at the release bar
0.95 (truth_v13); overspend voids the claim. The board's Justified is Supported, rung-3 facts on rung-1 truth, on the
agent's own loops that judge both limits (the free reference loop does not count):
  restrict  Supported if those loops include one of the world's SUPPORT_K nearest routes (every route ranked by true
            P(all 3), ties broken by the median joint ratio, so with no passing route they are its nearest misses)
            and every such loop reads failing (P(pass) <= 0.5);
  nominate  Supported by v1.2's rule below (Justified), unchanged.
Both = Outcome and Supported; none and no claim are never Supported. K = 100 is pre-registered
(TRACE_MINING_PLAN.md); 30 and 200 are reported as sensitivity. The rung-2 observer (justify_v13.py) is a diagnostic
and never ranks. v1.2's restriction rule is reported as rule_justified only. credits_to_supported scores the final
claim on the record cut at each credit checkpoint. The pool-posterior diagnostic is off. p_pass reads the knee with
its v1.3 error. The v1.2 text follows for the record.

Route claim (CONTRACT.md):
  Outcome (rung 1)   nominate: the route's P(all 3 bodies of a fresh preparation meet 10 kOe / 11 MGOe) >= 0.5 on owner
                     confirmation lots (sdl2_v12.sequential_p_all: 64-256 lots, the estimator world_truth uses); restrict: no declared route meets it in this world (world_truth); none and
                     a missing or malformed claim: false. A claim fails if the ledger's spend exceeds the allowance.
  Justified (rung 2)  an evidence rule anchored in the world, on the agent's own loops (the free reference loop does not
                     count): restrict is justified if the agent loop-measured one of the world's best routes (median
                     joint margin within sdl2_v12.best_route_set's tolerance of the best route) and at least one such
                     loop reads as failing (P(pass) <= 0.5 under the declared 2% log reading model); nominate is
                     justified if loops on at least NOMINATE_BODIES distinct bodies of the exact route, at its own stage,
                     all pass with P >= 0.95 under that model. A loop must include bhmax_MGOe or the curve to judge;
                     an own Hci-only loop on the route does not count toward the bodies but must read P(Hci >= 10) >= 0.95.
                     The importance-sampled posterior over the evaluator pool (P(some route meets)) is reported as a
                     diagnostic when its effective sample size allows (it collapses after a few loops; see README).
                     Pool rows hold strata inputs, not every passing route, so no per-route posterior is reported.
  Both               Outcome and Justified.
Forecasts (a diagnostic, not a reward): every frozen numeric forecast against the returned reading: 90% interval score
and log score, on the log scale for magnetic quantities; carry-forward baseline = the episode's latest earlier reading of
the same quantity with the instrument's declared spread. Abstentions and omissions are counted; skill is reported over
scored forecasts and over every returned reading (abstained or omitted ones at zero skill).
Discovery: whether the agent loop-measured a passing route at its own stage (rungs 1+3); a true withhold is a passing
reading (P >= 0.95) on a truly passing route followed by a restriction or none.
Decision reward (proposed for training, not used here; rung-1 anchored): +1 correct nomination or a correct and justified
restriction; 0 for none or a correct but unjustified restriction; -t/(1-t) = -1 at the declared t = 0.5 for a wrong
restriction or a failing nomination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sqlite3
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ndfeb_sps import economy_v12 as economy
from ndfeb_sps.models import c7_campaign as c7, sdl2_v11 as v11, sdl2_v12 as v12

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "sdl2-v1.2-20261026/worlds"))
import worlds  # noqa: E402
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdl2_v13 as v13  # noqa: E402
import truth_v13 as T13  # noqa: E402

VERSION = "sdl2-v1.3-grader.2"
RESTRICT_CUT, NOMINATE_CUT, T_PENALTY = 0.20, 0.50, 0.5
PASS_SUPPORT, NOMINATE_BODIES = 0.95, 2
SUPPORT_K, SUPPORT_K_REPORTED = 100, (30, 100, 200)   # pre-registered Oct 29 (TRACE_MINING_PLAN.md); 30, 200 sensitivity
CHECKPOINTS = (114, 228, 342, 456)   # credits: 20% steps of the 570-credit allowance
LOOP_QUANTITIES = ("hci_kOe", "br_kGs", "bhmax_MGOe", "demagnetization_curve")
TEMPER = {"density_g_cm3": 0.08, "log_hci": 0.10, "log_br": 0.10}   # declared evaluator likelihood widths
MIN_ESS = 20.0
TOOL_STAGE = {"as_sintered": "as_sintered", "after_first_anneal": "after_900", "after_second_anneal": "after_900_520"}
SPREAD = {"density_g_cm3": None, "hci_kOe": 0.02, "br_kGs": 0.02, "bhmax_MGOe": 0.05,
          "hci_screen_kOe": 0.02}   # carry-forward spreads (declared reading model)
Z90 = NormalDist().inv_cdf(0.95)


def normal_logpdf(mu, sd, y):
    return -0.5 * ((y - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


# ------------------------------------------------------------------ episode records

def records(ep):
    cfg = json.loads((ep / "simulation-magnetic/reference_world.json").read_text())
    contexts, native = [], {}
    for f in sorted((ep / "simulation-magnetic/executions").glob("*/native/*/*.json")):
        x = json.loads(f.read_text())
        if isinstance(x, dict) and x.get("provenance", {}).get("world_version") == v13.VERSION:
            native[x["delivery_id"]] = x
        elif isinstance(x, dict) and set(x) == {"request", "world_config", "seed"}:
            if x["world_config"] != cfg:
                raise ValueError("executed context differs from the episode world")
            contexts.append(x)
    bound = {hashlib.sha256(c7._canonical({"request": c["request"], "world": cfg, "seed": c["seed"]})).hexdigest(): c
             for c in contexts}
    db = f"file:{ep / 'simulation-magnetic/campaign/campaign.sqlite3'}?mode=ro&immutable=1"
    observations, executed, entries = [], [], []
    with sqlite3.connect(db, uri=True) as conn:
        requests = {json.loads(d)["request_id"]: json.loads(d)
                    for (d,) in conn.execute("SELECT data FROM objects WHERE kind='request' ORDER BY created_at, id")}
        delivered = set()
        for (d,) in conn.execute("SELECT data FROM objects WHERE kind='delivery' ORDER BY created_at, id"):
            delivery = json.loads(d)
            nat = native.get(delivery["delivery_id"])
            if nat is None:
                continue
            ctx = bound.get(nat["provenance"]["request_world_seed_sha256"])
            if ctx is None:
                raise ValueError("delivery not bound to an executed context")
            raw = {r["observation_id"]: r for r in nat["observations"]}
            for row in delivery["observations"]:
                if any(row.get(k) != v for k, v in raw[row["observation_id"]].items()):
                    raise ValueError("measurement changed between execution and ledger")
                observations.append({**row, "request_id": delivery["request_id"]})
            executed.append({"request_id": delivery["request_id"], "request": ctx["request"]})
            delivered.add(delivery["request_id"])
    for rid, rec in requests.items():
        entries.append({"request_id": rid, "request": rec["request"], "delivered": rid in delivered,
                        "prediction": rec.get("prediction"), "committed_at": rec.get("committed_at")})
    return cfg, observations, executed, entries


# ------------------------------------------------------------------ evaluator posterior (rung 2)

def evidence_summary(observations):
    """Independent readings: geometric/saturated densities and one (Hci, Br) pair per loop acquisition."""
    dens, loops = [], {}
    for o in observations:
        key = (o["specimen_id"], o.get("piece_id"), TOOL_STAGE.get(o.get("process_stage")))
        if o["quantity"] == "density_g_cm3" and o.get("density_method") in ("geometric", "archimedes_saturated"):
            dens.append((key, float(o["value"])))
        elif o["quantity"] in ("hci_kOe", "br_kGs", "bhmax_MGOe", "demagnetization_curve"):
            lr = o.get("loop_reading") or {}
            slot = loops.setdefault(o.get("loop_id") or o["observation_id"], {"key": key})
            for q in ("hci_kOe", "br_kGs"):
                if isinstance(lr.get(q), (int, float)):
                    slot[q] = float(lr[q])
    return {"density": dens, "loops": [l for l in loops.values() if "hci_kOe" in l and "br_kGs" in l]}


_POOL = {}


def _init(executed, summary):
    _POOL.update(executed=executed, summary=summary)


def candidate_loglik(world_config):
    try:
        W = c7.World(world_config["base_world"])
        truth = {}
        for e in _POOL["executed"]:
            for sid, ctx, piece, steps, state in v11.request_states(e["request"], W):
                vals, density = v12.bounded_state(W, state, sid, piece, steps, path=world_config["magnetic_path"])
                truth[(sid, piece, state["tool_stage"])] = (vals, density)
    except Exception:  # noqa: BLE001 - a candidate that cannot execute the agent's record has zero likelihood
        return -math.inf
    ll = 0.0
    for key, value in _POOL["summary"]["density"]:
        if key not in truth:
            return -math.inf
        ll += -0.5 * ((value - truth[key][1]) / TEMPER["density_g_cm3"]) ** 2
    for loop in _POOL["summary"]["loops"]:
        if loop["key"] not in truth:
            return -math.inf
        vals = truth[loop["key"]][0]
        ll += -0.5 * (math.log(loop["hci_kOe"] / vals["hci_kOe"]) / TEMPER["log_hci"]) ** 2
        ll += -0.5 * (math.log(loop["br_kGs"] / vals["br_kGs"]) / TEMPER["log_br"]) ** 2
    return ll


def design_weights(rows):
    top = worlds.supplied_top()[0]["id"]
    design = worlds.DESIGN
    shares = {}
    for feasible, classes in ((False, design["infeasible"]), (True, design["feasible"])):
        total = sum(classes.values())
        for cls, n in classes.items():
            shares[(feasible, cls)] = 0.5 * n / total
    labels = [(r["feasible"], worlds.strata(r, top)["class"]) for r in rows]
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return np.array([shares.get(lab, 0.0) / counts[lab] for lab in labels])


def posterior(executed, observations, procs=8, pool_rows=None):
    rows = pool_rows if pool_rows is not None else worlds.load("evaluator")
    rows = [r for r in rows if worlds.strata(r, worlds.supplied_top()[0]["id"])["admitted"]]
    w0 = design_weights(rows)
    summary = evidence_summary(observations)
    with mp.Pool(procs, initializer=_init, initargs=(executed, summary)) as p:
        ll = np.array(p.map(candidate_loglik, [r["world"] for r in rows], chunksize=16))
    finite = np.isfinite(ll) & (w0 > 0)
    if not finite.any():
        return {"assessable": False, "reason": "no candidate world reproduces the record", "ess": 0.0}
    logw = np.where(finite, np.log(np.where(w0 > 0, w0, 1.0)) + ll, -np.inf)
    w = np.exp(logw - logw[finite].max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    feasible = np.array([r["feasible"] for r in rows], float)
    out = {"assessable": ess >= MIN_ESS, "ess": ess, "candidates": int(finite.sum()),
           "p_some_route_meets": float(np.sum(w * feasible)),
           "readings": {"densities": len(summary["density"]), "loops": len(summary["loops"])},
           "temper": TEMPER, "prior": "panel design strata over the evaluator pool"}
    out["_weights"], out["_rows"] = w, rows
    return out


# ------------------------------------------------------------------ forecasts (rung 1)

def interval_score(lo, hi, y, alpha=0.10):
    return (hi - lo) + (2 / alpha) * max(lo - y, 0.0) + (2 / alpha) * max(y - hi, 0.0)


def score_one(quantity, mean, sd, y):
    if quantity == "density_g_cm3":
        lo, hi = mean - Z90 * sd, mean + Z90 * sd
        log = normal_logpdf(mean, sd, y) if sd > 0 else -math.inf
        return interval_score(lo, hi, y), log
    s = sd / mean if mean > 0 else math.inf
    if not (mean > 0 and y > 0 and s > 0 and math.isfinite(s)):
        return math.inf, -math.inf
    lo, hi = math.log(mean) - Z90 * s, math.log(mean) + Z90 * s
    return interval_score(lo, hi, math.log(y)), normal_logpdf(math.log(mean), s, math.log(y))


def forecasts(entries, observations):
    """Frozen forecasts against returns. A forecast is scored only if numeric with sd > 0 (and mean > 0 for magnetic
    quantities) and its measurement_stage equals the observation's stage; otherwise its state says why. The
    carry-forward baseline is the latest reading of the same quantity from an earlier request."""
    order = {e["request_id"]: i for i, e in enumerate(entries)}
    by_req = {}
    for o in observations:
        by_req.setdefault(o["request_id"], []).append(o)
    seen, rows = {}, []
    counts = {k: 0 for k in ("numeric", "abstained", "omitted", "numeric_without_sd", "stage_mismatch", "invalid")}
    for e in sorted(entries, key=lambda e: order[e["request_id"]]):
        obs = by_req.get(e["request_id"], [])
        preds = {p["specimen_id"]: p for p in ((e.get("prediction") or {}).get("predictions") or [])
                 if isinstance(p, dict) and "specimen_id" in p}
        latest = {}
        for o in obs:
            q = o["quantity"]
            y = float(o["value"]) if q in SPREAD and isinstance(o["value"], (int, float)) else None
            if y is None:
                continue
            if e["request_id"] != "REF-INITIAL":
                p = (preds.get(o["specimen_id"]) or {}).get(q)
                base = seen.get(q)
                row = {"request_id": e["request_id"], "specimen_id": o["specimen_id"], "quantity": q, "observed": y}
                magnetic = q != "density_g_cm3"
                if not isinstance(p, dict):
                    state = "omitted"
                elif p.get("mean") is None:
                    state = "abstained"
                elif not (isinstance(p.get("sd"), (int, float)) and p["sd"] > 0):
                    state = "numeric_without_sd"
                elif p.get("measurement_stage") != o.get("process_stage"):
                    state = "stage_mismatch"
                elif not isinstance(p.get("mean"), (int, float)) or (magnetic and p["mean"] <= 0):
                    state = "invalid"
                else:
                    state = "numeric"
                    row.update(mean=p["mean"], sd=p["sd"])
                    row["is90"], row["log_score"] = score_one(q, float(p["mean"]), float(p["sd"]), y)
                counts[state] += 1
                row["state"] = state
                if base is not None:
                    bsd = base[1] if SPREAD[q] is None else SPREAD[q] * base[0]
                    row["carry_forward"] = {"value": base[0], "sd": bsd}
                    row["cf_is90"], row["cf_log_score"] = score_one(q, base[0], bsd, y)
                rows.append(row)
            sd = (o.get("uncertainty") or {}).get("reading_sd_g_cm3")
            latest[q] = (y, float(sd) if isinstance(sd, (int, float)) and sd > 0 else 0.1)
        seen.update(latest)   # readings from this request become the baseline only for later requests
    scored = [r for r in rows if r["state"] == "numeric" and "cf_is90" in r and math.isfinite(r["is90"])]
    summary = {}
    for q in SPREAD:
        sub = [r for r in scored if r["quantity"] == q]
        # skill over every returned reading with a carry-forward baseline: an abstained, omitted or unscorable forecast
        # counts as carry-forward (zero skill), so abstaining never raises the mean (a diagnostic; not a reward)
        every = [r for r in rows if r["quantity"] == q and "cf_is90" in r]
        if sub:
            summary[q] = {"n": len(sub), "n_returns": len(every), "mean_is90": float(np.mean([r["is90"] for r in sub])),
                          "mean_cf_is90": float(np.mean([r["cf_is90"] for r in sub])),
                          "skill_is90": float(np.mean([r["cf_is90"] - r["is90"] for r in sub])),
                          "skill_is90_all_returns": float(sum(r["cf_is90"] - r["is90"] for r in sub) / len(every))}
    return {"counts": counts, "by_quantity": summary, "rows": rows}


# ------------------------------------------------------------------ claims

def parse_decision(d):
    claim = d.get("claim") if isinstance(d, dict) else None
    if not isinstance(claim, dict) or claim.get("form") not in ("nominate", "restrict", "none"):
        return None, "claim.form must be nominate, restrict or none"
    return claim, None


def read_decision(ep):
    path = ep / "pi/workspace/decision.json"
    if path.is_symlink():   # the workspace is agent-writable: never follow a link out of it
        return None, "decision.json is a symbolic link"
    if not path.exists():
        return None, "decision.json missing"
    try:
        d = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError, OSError) as e:
        return None, f"decision.json unreadable: {e}"
    return parse_decision(d)


def measured_loops(executed, observations):
    """Every loop acquisition with the declared route its body was in at that stage (None if off the set-points)."""
    loops = {}
    for o in observations:
        if o["quantity"] in LOOP_QUANTITIES:
            slot = loops.setdefault((o["request_id"], o["specimen_id"], o.get("loop_id")), {})
            lr = o.get("loop_reading") or {}
            slot.update({k: float(v) for k, v in lr.items() if isinstance(v, (int, float))})
    out = []
    for i, e in enumerate(executed):
        state = economy.ledger([{"request_id": x["request_id"], "request": x["request"], "delivered": True}
                                for x in executed[:i + 1]])
        for (rid, sid, lid), reading in loops.items():
            if rid != e["request_id"] or sid not in state["bodies"]:
                continue
            route = economy.declared_route(state["bodies"][sid], state["preparations"])
            out.append({"request_id": rid, "specimen_id": sid, "loop_id": lid, "reading": reading,
                        "route": v12.route_id(route) if route else None,
                        "own": rid not in economy.FREE_REQUESTS})
    return out


def p_pass(reading, bhmax, n=12000):
    """P(Hci >= 10 kOe and (BH)max >= 11 MGOe) for one loop under the declared 2% log reading model (log-flat prior)."""
    if not all(isinstance(reading.get(k), float) and reading[k] > 0 for k in ("hci_kOe", "br_kGs", "hk_over_hci")):
        return None
    rng = np.random.default_rng(11026)
    hc = rng.lognormal(math.log(reading["hci_kOe"]), 0.02, n)
    br = rng.lognormal(math.log(reading["br_kGs"]), 0.02, n)
    # v1.3: the knee field is read with error (sdl2_v13._loop), so the true Hk/Hci is the knee field over the true Hci
    hk = rng.lognormal(math.log(reading["hk_over_hci"] * reading["hci_kOe"]),
                       math.hypot(v13.HK_ACQ_LOG_SD, v13.HK_PIECE_LOG_SD), n)
    bh = bhmax(br, hc, np.clip(hk / hc, 0.05, 0.95))
    return float(np.mean((hc >= v12.REQUIREMENT["hci_kOe"]) & (bh >= v12.REQUIREMENT["bhmax_MGOe"])))


def best_routes(truth):
    ratios = {r["id"]: v12.joint_ratio(r["median"]["hci_kOe"], r["median"]["bhmax_MGOe"]) for r in truth["routes"]}
    best, top, _ = v12.best_route_set(ratios)
    return best, top


def nearest_routes(truth):
    """Every route, nearest to passing first: true P(all 3) (truth_v13), ties broken by the median joint ratio."""
    key = lambda r: (r["p_all"], v12.joint_ratio(r["median"]["hci_kOe"], r["median"]["bhmax_MGOe"]))  # noqa: E731
    return [r["id"] for r in sorted(truth["routes"], key=key, reverse=True)]


def discovery(cfg, truth, loops):
    """The agent's own loops only (the free reference loop is context, not the agent's evidence)."""
    passing = set(truth["passing"])
    own = [l for l in loops if l["own"] and l["route"]]
    built = {l["route"] for l in own}
    seen_pass = {l["route"] for l in own if l["route"] in passing and l["p_pass"] is not None and l["p_pass"] >= PASS_SUPPORT}
    best, best_ratio = best_routes(truth)
    return {"routes_loop_measured": sorted(built), "passing_route_measured": bool(built & passing),
            "measured_passing_routes": sorted(built & passing), "passing_reading_on_passing_route": bool(seen_pass),
            "best_route_measured": bool(built & best), "n_best_routes": len(best), "best_median_ratio": best_ratio}


def evaluate(cfg, observations, executed, entries, claim, error=None, *, procs=8, pool_rows=None, truth=None, post=None,
             diagnostic_posterior=False):
    """Score one record. truth and post may be supplied (fixtures that share a world or an evidence set)."""
    spend = economy.ledger(entries)
    truth = truth or T13.world_truth(cfg)
    form = claim["form"] if claim else "no_claim"
    result = {"version": VERSION, "claim_form": form, "claim_error": error,
              "spend": {k: spend[k] for k in ("credits", "new_bodies", "counts", "within_allowance")},
              "world": {"feasible": truth["feasible"], "n_passing_routes": len(truth["passing"]),
                        "magnetic_path": cfg["v12"]["magnetic_path"]}}
    if post is None and diagnostic_posterior:
        post = posterior(executed, observations, procs=procs, pool_rows=pool_rows)
    if post is not None:
        result["posterior_diagnostic"] = {k: v for k, v in post.items() if not k.startswith("_")}
    bhmax = c7.World(cfg["v12"]["base_world"]).link._bhmax
    loops = measured_loops(executed, observations)
    for l in loops:
        l["p_pass"] = p_pass(l["reading"], bhmax)
        h = l["reading"].get("hci_kOe")
        l["p_hci"] = (NormalDist().cdf(math.log(h / v12.REQUIREMENT["hci_kOe"]) / 0.02)
                      if isinstance(h, float) and h > 0 else None)   # P(Hci >= 10 kOe) under the 2 % log reading model
    best, _ = best_routes(truth)
    outcome = justified = None
    supported_k = {k: False for k in SUPPORT_K_REPORTED}   # none and no claim are never Supported
    if form == "nominate":
        try:
            route = v12.canonical_route(claim.get("route"))
            rid = v12.route_id(route)
            o = T13.route_outcome(cfg, route)
            outcome = o["meets"] and spend["within_allowance"]
            own = [l for l in loops if l["own"] and l["route"] == rid]
            # Only loops with (BH)max judge both limits and count toward the bodies; every own reading on the route,
            # an Hci-only loop included, must read as passing on what it measured.
            bodies = {l["specimen_id"] for l in own if l["p_pass"] is not None}
            consistent = all((l["p_pass"] if l["p_pass"] is not None else l["p_hci"]) >= PASS_SUPPORT
                             for l in own if l["p_pass"] is not None or l["p_hci"] is not None)
            justified = len(bodies) >= NOMINATE_BODIES and consistent
            supported_k = {k: justified for k in SUPPORT_K_REPORTED}   # v1.3: the nomination standard is unchanged
            result["nominated"] = {"route": rid, "p_all_truth": o["p_all"], "lots": o["lots"],
                                   "own_loops_on_route": [l["p_pass"] for l in own], "own_bodies_on_route": len(bodies)}
        except ValueError as e:
            outcome, justified, result["claim_error"] = False, False, f"nomination off the declared routes: {e}"
    elif form == "restrict":
        outcome = (not truth["feasible"]) and spend["within_allowance"]
        tested = [l["p_pass"] for l in loops if l["own"] and l["route"] in best]
        justified = any(p is not None and p <= 0.5 for p in tested)
        result["restriction_evidence"] = {"best_routes": len(best), "own_loops_on_best_routes": tested}
        rank = {rid: i for i, rid in enumerate(nearest_routes(truth))}
        judged = [l for l in loops if l["own"] and l["route"] in rank and l["p_pass"] is not None]
        for k in SUPPORT_K_REPORTED:
            on = [l["p_pass"] for l in judged if rank[l["route"]] < k]
            supported_k[k] = bool(on) and all(p <= 0.5 for p in on)
        result["restriction_evidence"]["nearest_rank_measured"] = min((rank[l["route"]] + 1 for l in judged), default=None)
    else:
        outcome = False
    supported = supported_k[SUPPORT_K]
    both = bool(outcome and supported)   # v1.3: Both = Outcome and Supported at the pre-registered K
    if form == "nominate":
        reward = 1.0 if outcome else -T_PENALTY / (1 - T_PENALTY)
    elif form == "restrict":
        reward = (1.0 if justified else 0.0) if outcome else -T_PENALTY / (1 - T_PENALTY)
    else:
        reward = 0.0
    disc = discovery(cfg, truth, loops)
    result.update(Outcome=outcome, Supported=supported, supported_by_K={str(k): v for k, v in supported_k.items()},
                  rule_justified=justified, Both=both,
                  wrong_restriction=form == "restrict" and truth["feasible"],
                  failed_nomination=form == "nominate" and not outcome,
                  true_withhold=form in ("restrict", "none", "no_claim") and disc["passing_reading_on_passing_route"],
                  discovery=disc, loops=[{k: l[k] for k in ("request_id", "specimen_id", "route", "p_pass", "own")}
                                         for l in loops],
                  forecasts=forecasts(entries, observations))
    return result


def prefix(observations, executed, entries, credits):
    """The record cut at a credit checkpoint, as v1.2's justify_modal/prefix_ledgers.py cuts it: deliveries in delivery
    order while the cumulative credits of the delivered requests (economy.ledger) stay at or below the checkpoint; the
    free requests are always kept, undelivered ones never."""
    seen, keep = {}, set()
    for e in executed:
        seen.setdefault(e["request_id"], e)
        if delivered_credits(seen.values()) <= credits or e["request_id"] in economy.FREE_REQUESTS:
            keep.add(e["request_id"])
    return ([o for o in observations if o["request_id"] in keep], [e for e in executed if e["request_id"] in keep],
            [e for e in entries if e["request_id"] in keep and e["delivered"]])


def delivered_credits(executed):
    return economy.ledger([{"request_id": x["request_id"], "request": x["request"], "delivered": True}
                           for x in executed])["credits"]


def credits_to_supported(cfg, observations, executed, entries, claim, truth):
    """Credits to a supported decision (TRACE_MINING_PLAN.md): the final claim scored on the record cut at each
    checkpoint; a checkpoint at or past the final spend is the final record. The value is the smallest checkpoint, the
    allowance included, from which the claim is Supported at every later one; None if the final record does not support
    it. `lost` flags a Supported checkpoint followed by one that is not."""
    final = delivered_credits({e["request_id"]: e for e in executed}.values())
    grid = CHECKPOINTS + (economy.ALLOWANCE["credits"],)
    end = evaluate(cfg, observations, executed, entries, claim, truth=truth)["Supported"]
    sup = {c: evaluate(cfg, *prefix(observations, executed, entries, c), claim, truth=truth)["Supported"]
           if c < final and c < grid[-1] else end for c in grid}   # the allowance point is always the end
    value = min((c for c in grid if all(sup[d] for d in grid if d >= c)), default=None) if end else None
    lost = any(sup[c] and not sup[d] for c in grid for d in grid if d > c)
    return {"final_credits": final, "supported_at": {str(c): v for c, v in sup.items()}, "credits_to_supported": value,
            "lost": lost}


def grade(ep, procs=8, pool_rows=None):
    ep = Path(ep)
    cfg, observations, executed, entries = records(ep)
    claim, error = read_decision(ep)
    result = evaluate(cfg, observations, executed, entries, claim, error, procs=procs, pool_rows=pool_rows)
    result["episode"] = ep.name
    (ep / "GRADE_V13.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("episode")
    a.add_argument("--procs", type=int, default=8)
    x = a.parse_args()
    r = grade(x.episode, x.procs)
    print(json.dumps({k: r.get(k) for k in ("claim_form", "Outcome", "Supported", "supported_by_K", "rule_justified", "Both",
                                        "wrong_restriction", "true_withhold", "discovery", "posterior_diagnostic",
                                        "world", "spend")},
                     indent=1, default=float))
