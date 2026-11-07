"""SDL-2 world generator: candidate pools, family truth, strata, admission and panel selection. Owner only, CPU.

  pool BLOCK START N PROCS   compute the strata inputs (pool_summary) for seeds START..START+N-1 into
                             owner/pool_BLOCK.jsonl; full family truth is owner_world.world_truth (fixtures, grader)
  panel [PREFIX BLOCK SCALE] select a panel (default: dev from owner/pool_dev.jsonl at DESIGN x 1) into owner/; a
                             training panel uses the same DESIGN from its own pool block (e.g. panel train train 4)
  summary                    print pool strata counts

Seed blocks are disjoint from every seed used before. The sealed test block is not drawn here: it is
drawn once, after the training recipe and checkpoint-selection rule are frozen.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import multiprocessing as mp
import random
import sys
from collections import Counter
from pathlib import Path

from ndfeb_sps.models import owner_world as v

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / "owner"
PKG = Path("/home/jarrodbarnes/ndfeb-sps-campaign/artifacts/studies/material-lab-20261019/instruments/C7.3")
MEAS = {"density_body_sd_g_cm3": 0.02, "link_model_draws": 4000}   # the live twin's measurement block
LOT = "SDL2-LOT"
BLOCKS = {"dev": 7_300_001, "evaluator": 7_400_001, "train": 7_500_001}
USED_SEEDS = ["2026102401 (development case, fixtures, diagnostic)", "7000001-7000600 (world scan)",
              "7290001-7290300 (pilot)", "3636691073607284804 (live lab twin)"]
NEAR_MISS = 0.85          # infeasible world whose best route's median body reaches >= 85% of both bars
BORDERLINE = (0.35, 0.65)  # feasible-or-not is weakly identified even with unlimited bodies of the best route
# The live program's route and planned endpoint (the reference body's SPS route, annealed 900 C / 2 h + 520 C / 2 h)
CANONICAL = "fitted|4.0|850.0|300.0|10.0|60.0|on_cooling|None|compact|900_2h+520"
TOP_K = 200               # routes kept per pool row (highest median joint ratio first)
# Lever groups: which controls separate a feasible world's nearest passing route from the live route
GROUPS = {"preparation": "preparation", "feedstock_d50_um": "feedstock", "temperature_c": "temperature_or_hold",
          "hold_time_s": "temperature_or_hold", "heating_rate_c_min": "rate_or_pressure",
          "pressure_mpa": "rate_or_pressure", "pressure_schedule": "load", "pressure_full_at_c": "load",
          "current_path": "current_path", "anneal": "anneal"}

DESIGN = {  # development panel: 48 decision states (re-derived on the expanded routes)
    # Infeasible lots split by whether the supplied model's top route is among the world's best routes (the routes a
    # justified restriction must have measured); the panel holds it to half.
    "infeasible": {"near_miss_model_best": 6, "near_miss_best_elsewhere": 6, "clear_model_best": 6,
                   "clear_best_elsewhere": 6},
    # The supplied model's own top route passes in only some feasible lots (held to 6 of 24 so deferring to it cannot
    # win); the rest are split by the levers that separate the nearest passing route from the live route.
    # Derived from the 4,000-world development pool (owner/POOL_STRATA.json): feasible classes multi 121, feedstock 43,
    # temperature_or_hold 28, canonical 10, rate_or_pressure 8, anneal 5, load 4, preparation 2, current_path 0.
    "feasible": {"model_top_passes": 6, "multi": 4, "feedstock": 4, "temperature_or_hold": 3, "canonical_passes": 2,
                 "rate_or_pressure": 2, "anneal": 1, "load": 1, "preparation": 1},
    "max_borderline": 9}


def ratio(median):
    return min(median["hci_kOe"] / v.REQUIREMENT["hci_kOe"], median["bhmax_MGOe"] / v.REQUIREMENT["bhmax_MGOe"])


def _ids():
    routes = v.declared_routes()
    return routes, [v.route_id(r) for r in routes]


def pool_summary(world, top_id):
    """The strata inputs of one world, each exactly as the full world_truth would give it, with P(all 3) estimated
    (owner_world.sequential_p_all) only where it can change them:
      feasible, best, best_p_all  screened routes in descending median joint ratio until one has P(all 3) = 1 (the
                                  highest-ratio route with the maximum estimate is world_truth's best); otherwise all;
      lps_only                    in an LPS world, also screened routes that do not rest on LPS from below 85 %TD;
      top / canonical passes      those two routes;
      nearest_passing             screened routes in declared order at increasing number of controls changed from the
                                  live route, up to the first passing one (the first passing route at minimal change).
    Medians come from owner_world.median_table; the TOP_K highest-ratio medians are kept."""
    base, path = world["base_world"], world["magnetic_path"]
    routes, ids = _ids()
    meds = v.median_table(world, routes)
    W = v.campaign.World(base)
    salt = v.owner_salt(world)
    sampled = {}

    def p_all(i):
        if ids[i] not in sampled:
            if not v.meets(meds[i], v.SCREEN_MARGIN):
                return 0.0
            k, n = v.sequential_p_all(W, routes[i], path=path, salt=salt)
            sampled[ids[i]] = [k / n, n]
        return sampled[ids[i]][0]

    ratios = [ratio(m) for m in meds]
    order = sorted(range(len(routes)), key=lambda i: -ratios[i])
    screened = [i for i in order if v.meets(meds[i], v.SCREEN_MARGIN)]
    best_i, best_key = order[0], (0.0, ratios[order[0]])
    for i in screened:
        p = p_all(i)
        if (p, ratios[i]) > best_key:
            best_i, best_key = i, (p, ratios[i])
        if p >= 1.0:
            break
    feasible = best_key[0] >= v.REQUIREMENT["p_all"]
    lps = lambda i: bool(routes[i]["anneal"] != "none" and W.amech == "LPS"  # noqa: E731
                         and meds[i]["rho_made_rel"] < v.LPS_START_FLOOR_REL)
    lps_only = False
    if feasible and W.amech == "LPS":
        lps_only = not any(p_all(i) >= v.REQUIREMENT["p_all"] for i in screened if not lps(i))
    index = {rid: i for i, rid in enumerate(ids)}
    top_passes = p_all(index[top_id]) >= v.REQUIREMENT["p_all"]
    canonical_passes = p_all(index[CANONICAL]) >= v.REQUIREMENT["p_all"]
    nearest = None
    if feasible:
        canon = routes[index[CANONICAL]]
        dist = lambda i: sum(routes[i][k] != canon[k] for k in v.ROUTE_KEYS)  # noqa: E731
        for d in range(1, len(v.ROUTE_KEYS) + 1):
            hit = next((i for i in sorted(screened) if dist(i) == d and p_all(i) >= v.REQUIREMENT["p_all"]), None)
            if hit is not None:
                nearest = {"id": ids[hit], "changed": [k for k in v.ROUTE_KEYS if routes[hit][k] != canon[k]]}
                break
    best_set, best_ratio, tol = v.best_route_set(dict(zip(ids, ratios)))
    return {"feasible": feasible, "best": ids[best_i], "best_p_all": best_key[0], "best_ratio": max(ratios),
            "sampled": sampled, "n_screened": len(screened), "lps_only": lps_only,
            "max_raw_hci_kOe": max(m["raw_hci_kOe"] for m in meds), "top_passes": top_passes,
            "canonical_passes": canonical_passes, "nearest_passing": nearest, "top_in_best_set": top_id in best_set,
            "n_best_set": len(best_set), "best_set_tolerance": tol,
            "top_medians": [[ids[i], round(meds[i]["hci_kOe"], 4), round(meds[i]["bhmax_MGOe"], 4)]
                            for i in order[:TOP_K]]}


def one(seed_block):
    seed, block = seed_block
    world = v.draw(PKG, seed, f"v12-{block}-{seed}", MEAS, lot_id=LOT, drawn_at="2026-10-26")
    top = supplied_top()[0]["id"]
    return {"seed": seed, "block": block, "world": world, "path": world["magnetic_path"],
            "draw": world["base_world"]["draw"], "model_top": top, **pool_summary(world, top)}


def pool(block, start, n, procs):
    OWNER.mkdir(exist_ok=True)
    out = OWNER / f"pool_{block}.jsonl"
    done = {json.loads(line)["seed"] for line in out.open()} if out.exists() else set()
    todo = [(s, block) for s in range(start, start + n) if s not in done]
    with mp.Pool(procs) as p, out.open("a") as f:
        for row in p.imap_unordered(one, todo, chunksize=2):
            f.write(json.dumps(row) + "\n")
            f.flush()


_TOP = None


def supplied_top():
    """The supplied model's ranking of the declared routes, best first; ties go to the earlier declared route."""
    global _TOP
    if _TOP is None:
        order = {v.route_id(r): i for i, r in enumerate(v.declared_routes())}
        rows = {}
        for f in sorted(HERE.glob("supplied_ranking_*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                if r["id"] in order:
                    rows[r["id"]] = r
        assert len(rows) == len(order), "supplied-model ranking incomplete"
        rows = list(rows.values())
        for r in rows:
            r["p_mean"] = 0.5 * (r["link"]["p_all"] + r["c5"]["p_all"])   # paths at the generator's equal odds
        rows.sort(key=lambda r: (-r["p_mean"], order[r["id"]]))
        _TOP = rows
    return _TOP


def public_prior():
    """The disclosed BO prior: the supplied model's per-route rows, declared order, in a public
    file away from owner records."""
    order = {v.route_id(r): i for i, r in enumerate(v.declared_routes())}
    rows = sorted(supplied_top(), key=lambda r: order[r["id"]])
    out = HERE.parent / "arms/prior/supplied_ranking.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({k: r[k] for k in ("id", "route", "link", "c5")}) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "path": str(out), "top": supplied_top()[0]["id"]}))


def lever_class(row, top_id):
    """Which levers separate a feasible world's passing set from the defaults an agent would start from."""
    if row["top_passes"]:
        return "model_top_passes"
    if row["canonical_passes"]:
        return "canonical_passes"
    labels = {GROUPS[k] for k in row["nearest_passing"]["changed"]}
    return labels.pop() if len(labels) == 1 else "multi"


def strata(row, top_id):
    s = {"feasible": row["feasible"], "path": row["path"],
         "borderline": BORDERLINE[0] <= row["best_p_all"] <= BORDERLINE[1],
         # C5 coercivity on its 37.5 kOe asymptote is bounded in the world (owner_world.HCI_CEILING_KOE), not grounds to drop
         # the world: rejecting it removed 45% of the C5 path's feasible worlds. Flagged for disclosure.
         "admitted": not row["lps_only"], "saturating_route": row["max_raw_hci_kOe"] > v.HCI_CEILING_KOE}
    if row["model_top"] != top_id:
        raise ValueError("pool row computed against a different supplied-model top route")
    if row["feasible"]:
        s["class"] = lever_class(row, top_id)
    else:
        s["class"] = (("near_miss" if row["best_ratio"] >= NEAR_MISS else "clear")
                      + ("_model_best" if row["top_in_best_set"] else "_best_elsewhere"))
    return s


def load(block):
    return [json.loads(line) for line in (OWNER / f"pool_{block}.jsonl").open()]


def summary(block):
    """Strata of a pool (owner/POOL_STRATA.json, keyed by block): the counts the panel design is derived from."""
    top = supplied_top()[0]["id"]
    rows = load(block)
    st = [strata(r, top) for r in rows]
    adm = [(r, s) for r, s in zip(rows, st) if s["admitted"]]
    feas = [(r, s) for r, s in adm if r["feasible"]]
    infe = [(r, s) for r, s in adm if not r["feasible"]]
    out = {"block": block, "worlds": len(rows), "admitted": len(adm), "not_admitted_lps_only": len(rows) - len(adm),
           "feasible": len(feas), "feasible_share": round(len(feas) / max(len(adm), 1), 4),
           "model_top": top, "model_top_passes_among_feasible": round(sum(r["top_passes"] for r, _ in feas) / max(len(feas), 1), 4),
           "canonical_passes_among_feasible": round(sum(r["canonical_passes"] for r, _ in feas) / max(len(feas), 1), 4),
           "model_top_among_best_infeasible": round(sum(r["top_in_best_set"] for r, _ in infe) / max(len(infe), 1), 4),
           "borderline": sum(s["borderline"] for _, s in adm), "saturating_route": sum(s["saturating_route"] for _, s in adm),
           "classes": {"|".join(map(str, k)): n for k, n in sorted(Counter((r["feasible"], s["class"], r["path"]) for r, s in adm).items())},
           "nearest_passing_changed": dict(Counter(",".join(r["nearest_passing"]["changed"]) for r, _ in feas
                                                   if r["nearest_passing"]).most_common(25))}
    path = OWNER / "POOL_STRATA.json"
    allp = json.loads(path.read_text()) if path.exists() else {}
    allp[block] = out
    path.write_text(json.dumps(allp, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def panel_secret():
    """Owner-only key for public world hashes (an unsalted hash of a seed-determined world can be reversed by scanning
    the published seed block)."""
    path = OWNER / "PANEL_SECRET"
    if not path.exists():
        path.write_text(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
    return bytes.fromhex(path.read_text().strip())


def panel(prefix="dev", block="dev", scale=1):
    top = supplied_top()[0]["id"]
    rows = [r for r in load(block) if strata(r, top)["admitted"]]
    rng = random.Random(f"{prefix}-20261026")   # declared selection seed; selection never looks at agent behaviour
    rng.shuffle(rows)
    chosen, borderline = [], 0
    for feasible, classes in ((False, DESIGN["infeasible"]), (True, DESIGN["feasible"])):
        for cls, base_want in classes.items():
            want = base_want * scale
            # alternate the magnetic path so each class is close to path-balanced
            members = [r for r in rows if r["feasible"] == feasible and strata(r, top)["class"] == cls]
            by_path = {p: [r for r in members if r["path"] == p] for p in ("link", "c5")}
            picked = []
            while len(picked) < want and (by_path["link"] or by_path["c5"]):
                for p in ("link", "c5"):
                    if len(picked) < want and by_path[p]:
                        r = by_path[p].pop()
                        b = strata(r, top)["borderline"]
                        if b and borderline >= DESIGN["max_borderline"] * scale:
                            continue
                        borderline += b
                        picked.append(r)
            chosen += [(cls, r) for r in picked]
    feasible_short = sum(DESIGN["feasible"].values()) * scale - sum(1 for c, r in chosen if r["feasible"])
    if feasible_short > 0:   # fill from any feasible class not yet exhausted, so the reachable share stays at half
        taken = {r["seed"] for _, r in chosen}
        # fill with any other feasible class before model_top_passes, so a short class never raises the model-top share
        # above the design (the fill does not apply the borderline cap or path balance; panel() reports the classes)
        spare = sorted((r for r in rows if r["feasible"] and r["seed"] not in taken),
                       key=lambda r: strata(r, top)["class"] == "model_top_passes")
        chosen += [(strata(r, top)["class"], r) for r in spare[:feasible_short]]
    rng.shuffle(chosen)   # names carry no class or feasibility order
    key = panel_secret()
    states = []
    for i, (cls, r) in enumerate(chosen):
        name = f"{prefix}-{i + 1:02d}" if len(chosen) < 100 else f"{prefix}-{i + 1:04d}"
        world_json = json.dumps(r["world"], sort_keys=True).encode()
        states.append({"name": name, "seed": r["seed"], "class": cls, **strata(r, top), "best": r["best"],
                       "best_p_all": r["best_p_all"], "best_ratio": r["best_ratio"], "top_passes": r["top_passes"],
                       "canonical_passes": r["canonical_passes"], "nearest_passing": r["nearest_passing"],
                       "sampled": r["sampled"], "world": r["world"],
                       "world_sha256": hashlib.sha256(world_json).hexdigest(),
                       "public_id": hmac.new(key, world_json, hashlib.sha256).hexdigest()})
    OWNER.mkdir(exist_ok=True)
    (OWNER / f"{prefix.upper()}_PANEL.json").write_text(json.dumps(
        {"design": DESIGN, "scale": scale, "block": block, "model_top_route": top, "canonical": CANONICAL,
         "selection_seed": f"{prefix}-20261026", "states": states}, indent=1) + "\n")
    public = [{"name": s["name"], "public_id": s["public_id"]} for s in states]
    (HERE / f"{prefix.upper()}_PANEL_PUBLIC.json").write_text(json.dumps(public, indent=1) + "\n")
    print(json.dumps({"states": len(states), "feasible": sum(s["feasible"] for s in states),
                      "classes": Counter(s["class"] for s in states), "paths": Counter(s["path"] for s in states),
                      "borderline": sum(s["borderline"] for s in states)}, default=str, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pool":
        pool(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "panel":
        args = sys.argv[2:]
        panel(args[0] if args else "dev", args[1] if len(args) > 1 else "dev", int(args[2]) if len(args) > 2 else 1)
    elif cmd == "prior":
        public_prior()
    elif cmd == "summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else "dev")
