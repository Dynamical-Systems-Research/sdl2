"""SDL-2 v1.2 owner world: C7.3 process physics, dev.2 persistent loops, v1.2 physics corrections.

Owner only. The agent-facing physics model is the released C7.3 package, unchanged. This module is the hidden world
used by the virtual magnetic lab and by grading. It reuses C7.3's World (process, density, anneal, pieces) and the
dev.2 magnetometry (persistent hidden loop state, explicit acquisition identity, ideal own-piece normalization).

v1.2 changes, each documented in physics/PHYSICS_FIXES.md:
  1. Degassing completeness is an outcome of the executed program, not a control. C7 accepted a declared
     completeness (a number or "complete") because it has no degassing-kinetics model; "complete" zeroed residual
     hydrogen and, under the trap load-timing explanation, drove link Hci to 535 kOe. v1.2 executes only what the
     world models: method "none", or vacuum degassing at the fitted 600 C program (600-900 s).
  2. Coercivity is bounded at HCI_CEILING_KOE = 30 kOe. In some C5-path worlds coercivity sits on the path's own
     asymptote (37.5 kOe), for porous low-temperature bodies and for well-made ones; Dy-free sintered NdFeB reports reach
     about 20-25 kOe. The bound binds on those states, passing ones included. The 10 kOe requirement is far below it and
     (BH)max is Br-limited, so no world's feasibility changes; the worlds are kept and flagged (`saturating_route`).
  3. LPS extrapolation: a world whose feasibility rests only on pressureless liquid-assisted densification of a body
     below LPS_START_FLOOR_REL is rejected at admission (worlds/). The mechanism itself stays in the family: it is
     the live campaign's open hypothesis (#8's 7.07 -> 7.58).
The magnetic path (coercivity link or C5-Hci/Br) is a hidden per-world structural draw. The supplied C7.3 model
reports both paths, as it does for the live campaign.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math

import numpy as np

from .. import economy_v12 as economy
from . import c7_campaign as c7
from . import sdl2_v11 as v11

FAMILY = "ndfeb_sdl2_v12"
VERSION = "sdl2-v1.2-dev.1"   # owner-side only (private provenance); agent-visible rows use the neutral labels below
PUBLIC_WORLD, PUBLIC_READING = "sdl2-virtual-lot", "persistent-loop-acquisition"

# Owner-fixed intermediate requirement: SDL-2 v1.0's published bar, reachable for isotropic bodies in some worlds.
# The program objective (Hci >= 20 kOe and (BH)max >= 45 MGOe) needs alignment; no isotropic route reaches it.
REQUIREMENT = {"hci_kOe": 10.0, "bhmax_MGOe": 11.0, "bodies": 3, "p_all": 0.5, "measurement_temperature_C": 20.0}

# Declared routes: the public declaration in economy_v12 (one value of every control the world responds to;
# routes/ROUTE_EXPANSION.md). The world executes a route as these request conditions.
ROUTE_KEYS = economy.ROUTE_KEYS
DECLARED = economy.DECLARED
FIXED = economy.FIXED
ANNEAL_PROGRAMS = economy.ANNEAL_PROGRAMS
STAGES = economy.STAGE_OF_STEPS
PREPARATIONS = {"none": {"method": "none"},
                "fitted": {"method": "vacuum_degassing", "temperature_c": 600.0, "hold_time_s": 900.0}}

HCI_CEILING_KOE = 30.0
MIN_LOTS, MAX_LOTS, LOT_BATCH = 64, 256, 16
SCREEN_MARGIN = 0.97
LPS_START_FLOOR_REL = 0.85
FITTED_HOLD_S = (600.0, 900.0)
MEDIAN = ("OWNER-MEDIAN", "OWNER-MEDIAN", "OWNER-MEDIAN-B")   # lot, preparation and specimen of a route's median body


def declared_routes():
    return economy.declared_routes()


def route_id(route):
    return economy.route_id(route)


def canonical_route(route):
    """A declared route from a claimed route, or ValueError (off the declared set-points). A claim names every key;
    pressure_full_at_c may be omitted or null for the programs without a load-reached temperature."""
    if not isinstance(route, dict):
        raise ValueError(f"a route names {list(ROUTE_KEYS)}")
    route = dict(route)
    route.setdefault("pressure_full_at_c", None)
    if set(route) != set(ROUTE_KEYS):
        raise ValueError(f"a route names exactly {list(ROUTE_KEYS)}")
    out = {}
    for key in ROUTE_KEYS:
        value = route[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
        if value not in DECLARED[key]:
            raise ValueError(f"{key}={route[key]!r} is not a declared set-point {list(DECLARED[key])}")
        out[key] = value
    if not economy.admissible(out):
        raise ValueError("not a declared combination: pressure_full_at_c applies only with on_cooling, and insulated "
                         "only with on_heating or on_dwell")
    return out


def route_inputs(route):
    """(preparation conditions, manufacture conditions, executed anneal steps) of a declared route."""
    prep = dict(PREPARATIONS[route["preparation"]], feedstock_d50_um=route["feedstock_d50_um"])
    cond = {"temperature_c": route["temperature_c"], "heating_rate_c_min": route["heating_rate_c_min"],
            "hold_time_s": route["hold_time_s"], "pressure_mpa": route["pressure_mpa"],
            "pressure_schedule": route["pressure_schedule"], "current_path": route["current_path"], **FIXED}
    if route["pressure_full_at_c"] is not None:
        cond["pressure_full_at_c"] = route["pressure_full_at_c"]
    return prep, cond, list(ANNEAL_PROGRAMS[route["anneal"]])


# ------------------------------------------------------------------ physics correction 1: executed degassing

def validate_request(request):
    """v1.2 rejects declared degassing outcomes; everything else is C7.3's own validation."""
    for prep in request.get("preparations", []):
        cond = prep.get("conditions", {}) if isinstance(prep, dict) else {}
        if not isinstance(cond, dict):
            continue
        if "degassing_completeness" in cond or "degas_completeness" in cond:
            raise ValueError("degassing completeness is the outcome of the executed degassing program, not a "
                             "control; state method, temperature_c and hold_time_s only")
        if cond.get("method") == "vacuum_degassing":
            T, hold = cond.get("temperature_c"), cond.get("hold_time_s")
            if T != 600 or not isinstance(hold, (int, float)) or not FITTED_HOLD_S[0] <= hold <= FITTED_HOLD_S[1]:
                raise ValueError("not supported by this instrument: v1.2 has no degassing-kinetics model; it executes "
                                 "vacuum degassing at the fitted 600 C program (hold 600-900 s) or method 'none'. "
                                 "This is a limit of the instrument, not of the process")


# ------------------------------------------------------------------ physics correction 2: bounded coercivity

def _bound(world, values):
    if values["hci_kOe"] > HCI_CEILING_KOE:
        values = dict(values, hci_kOe=HCI_CEILING_KOE)
        values["bhmax_MGOe"] = float(world.link._bhmax(np.array([values["br_kGs"]]), np.array([values["hci_kOe"]]),
                                                       np.array([values["hk_over_hci"]]))[0])
    return values


def bounded_state(world, state, sid, piece, steps, *, path):
    """The v1.2 hidden values of one body state: the anneal hold's grain term on C5 coercivity (correction 4), then the
    dev.2 magnetic state, then the 30 kOe bound (correction 2)."""
    values, density = v11.magnetic_state(world, _h4_state(world, state), sid, piece, steps, path=path)
    return _bound(world, values), density


# ------------------------------------------------------------------ physics correction 4: the anneal hold on C5 Hci

def _h4_lh(world, r, lh, hold_h):
    """C5 ln Hci with the package's hold-scaled grain term (C7.3 predict()/joint_target(): _c7_h4_hold_shift through
    _c7_apply_lh_shift) for a world whose magnetic draw is the grain-size mechanism H4 and an annealed tool route r.
    c7.World.evaluate leaves C5's ln Hci at historical E's 2 h response for every hold; the package does not."""
    lh = np.asarray(lh, float)
    if r["anneal"] == "none" or world.mmech != "H4":
        return lh
    L = len(lh)
    shift = world.pm._c7_h4_hold_shift(r, np.full(L, world.mmech), np.full(L, world.midx), "post",
                                       np.broadcast_to(np.asarray(hold_h, float), (L,)))
    return world.pm._c7_apply_lh_shift(lh, shift)


def _h4_state(world, state):
    r = state["route"]
    if r["anneal"] == "none" or world.mmech != "H4":
        return state
    lh = float(_h4_lh(world, r, np.array([state["lh"]]), np.array([float(r["anneal_hold_h"])]))[0])
    return dict(state, lh=lh, hci_kOe=math.exp(lh))


def dev2_config(world_config):
    """The dev.2 configuration this world's loops and density readings are computed under."""
    return {"family": v11.FAMILY, "base_world": world_config["base_world"],
            "magnetic_path": world_config["magnetic_path"],
            "magnetic_log_sd": world_config.get("magnetic_log_sd", .02), "interface_version": v11.CORRECTED_VERSION}


def simulate_sdl2_v12(request, *, world_config, seed):
    """dev.2 execution with the v1.2 corrections. Reading identity, noise streams and C7.3 process physics unchanged;
    a magnetic reading uses the corrected hidden state (bounded_state: corrections 2 and 4)."""
    if world_config.get("family") != FAMILY:
        raise ValueError("wrong SDL-2 world family")
    validate_request(request)
    dev2 = dev2_config(world_config)
    delivery = v11.simulate_sdl2_v11(request, world_config=dev2, seed=seed)
    world = c7.World(world_config["base_world"])
    states = {sid: (context, piece, steps, state) for sid, context, piece, steps, state in v11.request_states(request, world)}
    world_key = hashlib.sha256(c7._canonical(dev2)).hexdigest()
    specs = {s["specimen_id"]: s for s in request["specimens"]}
    sigma = float(dev2["magnetic_log_sd"])
    for row in delivery["observations"]:
        # Agent-visible rows carry no version names (one public release): neutral labels for dev.2's and this world's
        # identifiers. The owner version stays in the delivery's private provenance (grade_v12.records matches it).
        row["world_version"] = PUBLIC_WORLD
        if "evaluation_version" in row:
            row["evaluation_version"] = PUBLIC_READING
        if row.get("method") == "sdl2_v11_hypothetical_corrected_loop":
            row["method"] = "hypothetical_persistent_loop"
        sid, quantity = row["specimen_id"], row["quantity"]
        if quantity not in c7.MAGNETIC:
            continue
        context, piece, steps, state = states[sid]
        true, _ = v11.magnetic_state(world, state, sid, piece, steps, path=dev2["magnetic_path"])
        bounded, _ = bounded_state(world, state, sid, piece, steps, path=dev2["magnetic_path"])
        if bounded == true:
            continue   # no v1.2 correction applies: dev.2's reading stands
        # Same acquisition stream as dev.2; only the corrected hidden state differs.
        identity = {"world": world_key, "lot": request["lot_id"], "specimen": sid,
                    "preparation": specs[sid]["preparation_id"], "cycle": specs[sid]["cycle_id"],
                    "piece": piece, "stage": state["tool_stage"], "acquisition": context["field_set_id"]}
        key = hashlib.sha256(c7._canonical(identity)).hexdigest()
        rng = world.stream7(v11.CORRECTED_VERSION, "loop-reading", key)
        reading = dict(bounded)
        for q in ("hci_kOe", "br_kGs"):
            reading[q] *= math.exp(sigma * float(rng.standard_normal()))
        reading["bhmax_MGOe"] = float(world.link._bhmax(np.array([reading["br_kGs"]]), np.array([reading["hci_kOe"]]),
                                                        np.array([reading["hk_over_hci"]]))[0])
        row["value"] = reading if quantity == "demagnetization_curve" else reading[c7.LINK_SOURCE[quantity]]
        row["loop_reading"] = (reading if quantity in ("demagnetization_curve", "bhmax_MGOe")
                               else {c7.LINK_SOURCE[quantity]: reading[c7.LINK_SOURCE[quantity]]})
    # Bind the delivery to this episode's v1.2 world (the bridge verifies request + world + seed), as dev.2 does for its own.
    delivery["provenance"].update(world_version=VERSION, world_family=FAMILY,
                                  request_world_seed_sha256=hashlib.sha256(c7._canonical(
                                      {"request": request, "world": world_config, "seed": seed})).hexdigest())
    return delivery


def truth_for(request, world_config):
    """Owner-only bounded values for the states in an executed request (the grader's hidden truth)."""
    world = c7.World(world_config["base_world"])
    rows = {}
    for sid, context, piece, steps, state in v11.request_states(request, world):
        values, density = bounded_state(world, state, sid, piece, steps, path=world_config["magnetic_path"])
        rows[(sid, piece, state["tool_stage"])] = {**values, "density_g_cm3": density,
                                                   "rho_made_rel": state["rho_body_rel"],
                                                   "rho_stage_rel": state["rho_stage_rel"]}
    return rows


# ------------------------------------------------------------------ route truth

class MedianWorld(c7.World):
    """The same world with every preparation, body and piece effect at zero: a route's median body."""

    def body(self, specimen_id):
        return {"rho_dev_g_cm3": 0.0, "u_spec": 0.0}

    def prep_z(self, lot_id, preparation_id):
        return 0.0

    def piece_effect(self, specimen_id, piece_id):
        return 0.0


def body_values(world, route, *, path, lot_id, preparation_id, specimen_id, piece_id="P1"):
    """Bounded hidden values of one body made and annealed by a declared route: the scalar reference path (c7.World
    body_state, sdl2_v11 magnetic_state, the 30 kOe bound). The batched paths below equal it body by body."""
    prep, cond, steps = route_inputs(route)
    controls = c7._manufacture_controls(cond, specimen_id)
    executed = c7._preparation(prep, preparation_id)
    state = world.body_state(lot_id, preparation_id, specimen_id, controls, executed, STAGES[len(steps)],
                             c7._first_hold_h(steps))
    raw, density = v11.magnetic_state(world, _h4_state(world, state), specimen_id, piece_id, steps, path=path)
    values = _bound(world, raw)
    return {"hci_kOe": values["hci_kOe"], "br_kGs": values["br_kGs"], "bhmax_MGOe": values["bhmax_MGOe"],
            "hk_over_hci": values["hk_over_hci"], "density_g_cm3": density, "raw_hci_kOe": raw["hci_kOe"],
            "rho_made_rel": state["rho_body_rel"], "rho_stage_rel": state["rho_stage_rel"]}


class _TiledRNG:
    """A generator whose every draw of size n is one draw of size 1 from `rng`, repeated n times: n bodies or routes
    evaluated together consume exactly the stream one evaluation consumes alone (the link's epistemic draws belong to
    the world, not to a body)."""

    def __init__(self, rng, n):
        self.rng, self.n = rng, n

    def _one(self, size):
        if size != self.n:
            raise ValueError("tiled draw of an unexpected size")
        return 1

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.repeat(self.rng.normal(loc, scale, self._one(size)), self.n)

    def standard_normal(self, size=None):
        return np.repeat(self.rng.standard_normal(self._one(size)), self.n)

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.repeat(self.rng.uniform(low, high, self._one(size)), self.n)

    def beta(self, a, b, size=None):
        return np.repeat(self.rng.beta(a, b, self._one(size)), self.n)

    def choice(self, a, size=None, replace=True, p=None):
        return np.repeat(self.rng.choice(a, size=self._one(size), replace=replace, p=p), self.n)


def _tool_route(world, controls, executed, anneal, hold_h=None):
    r = world.pm._route(c7.c7_route(controls, executed, anneal, hold_h))
    r.pop("_resolution", None)
    r["_degas_expected"] = world.pm._expected_state(r)
    return r


def _latent(world, r, rho_body, u_spec):
    """The package latent of c7.World.evaluate at tool route r for L bodies of known route-level density rho_body:
    degassing state, magnetic latent (sem), oxide factor. r carries no anneal hold the latent reads (the hold acts
    through _stage below)."""
    pm = world.pm
    L = len(rho_body)
    A = pm._arrays(r)
    state, eps = pm._degas_state(r, np.array([0.5]), world.stream5("world", "degas_extrapolation"))
    state = np.repeat(state, L)
    eps = None if eps is None else np.repeat(eps, L)
    mlab, midx = np.full(L, world.mmech), np.full(L, world.midx)
    lat = pm._magnetic_latent(A, r, mlab, midx, rho_body, state, np.full(L, world.z_hull), np.asarray(u_spec, float),
                              "post", eps=eps, indicators={"load_timing": np.full(L, world.lt)}, sem=True)
    rr = {k: v for k, v in r.items() if k != "_degas_expected"}
    ox = pm._c6_oxide_factor(rr, lat, mlab, midx, "post")
    v = {k: np.broadcast_to(np.asarray(x, float).reshape(-1), (L,)) for k, x in lat.items()}
    return lat, v, (None if ox is None else np.asarray(ox, float).reshape(-1))


def _stage(world, anneal_key, T_peak_C, lat, v, rho_body, hold_h, lh):
    """The anneal layer of c7.World.evaluate: stage density, Br and grain after the executed 900 C hold (hours, (L,));
    lh is C5's ln Hci at the state (correction 4 applied by the caller)."""
    pm, an = world.pm, world.an
    L = len(rho_body)
    steps = [] if anneal_key == "none" else [(900.0, np.asarray(hold_h, float))] + (
        [(520.0, np.array([2.0]))] if anneal_key == "900+520" else [])
    rho_stage = an.anneal_density(world.an_p, world.an_mech, rho_body, steps) if steps else rho_body
    br = np.exp(v["lbr"]) * ((rho_stage / rho_body) / _gain_E(world) if steps else 1.0)
    grain = v["grain"]
    if steps:
        grain = pm._c7_grain_hold(lat, np.full(L, world.mmech), np.full(L, world.midx), "post",
                                  np.broadcast_to(np.asarray(hold_h, float), (L,)), T_peak_C)
    return {"rho_stage_rel": rho_stage, "hci_kOe": np.exp(lh), "br_kGs": br, "grain_um": grain}


def _gain_E(world):
    """Historical E's densification gain over 900 C / 2 h under the world's anneal draw (c7.World.evaluate's gain_E)."""
    if "_v12_gain_E" not in world.__dict__:
        e = np.array([world.pm.fw.E_RHO])
        world._v12_gain_E = float(world.an.anneal_density(world.an_p, world.an_mech, e, [(900.0, np.array([2.0]))])[0]
                                  / e[0])
    return world._v12_gain_E


def _finish(world, tool_stage, made, state, eta, steps, *, path):
    """c6_state at the stages the link reads, sdl2_v11.magnetic_state and the 30 kOe bound, for L bodies or routes at
    one tool stage. made/state: (L,) arrays (rho_body_rel, W_as, grain_stage_um, oxide_factor, ln_chem_offset; grain_um,
    rho_stage_rel, hci_kOe, br_kGs). steps: executed steps with hold in seconds as (L,) arrays."""
    L = len(eta)
    ms, P = world.ms, world.c6
    g = made["grain_stage_um"]
    D = {"as_sintered": g["as_sintered"], "after_900": g["after_900_520"], "after_900_520": g["after_900_520"]}
    if tool_stage != "as_sintered":
        D[tool_stage] = state["grain_um"]
    a3 = {s: {k: np.broadcast_to(np.asarray(x, float).reshape(-1), (L,))
              for k, x in ms.a3_latent(P, made["W_as"], s, made["oxide_factor"]).items()}
          for s in {tool_stage, "as_sintered"}}
    rho = world.an.piece_density(world.an_p, world.an_mech, made["rho_body_rel"], eta,
                                 [(t, np.asarray(h, float) / 3600.0) for _, t, h in steps])
    full = lambda x: np.full(L, float(x))  # noqa: E731
    flat = {"grain_median_um": np.broadcast_to(np.asarray(D[tool_stage], float), (L,)),
            "grain_sigma_ln": full(P["c6_sig"][0]), "density_rel": rho,
            "nd_rich_area_fraction": a3[tool_stage]["nd_rich_area_fraction"],
            "pocket_share": a3[tool_stage]["pocket_share"], "faint_line_coverage": a3[tool_stage]["faint_line_coverage"],
            "pocket_share_as_sintered": a3["as_sintered"]["pocket_share"],
            "faint_line_coverage_as_sintered": a3["as_sintered"]["faint_line_coverage"],
            "coverage_sem_lower": np.full(L, np.nan), "stage": np.full(L, tool_stage),
            "alignment_cos_theta": full(world.tool.ALIGN_DEFAULT), "ln_hci_chem_offset": made["ln_chem_offset"]}
    core = getattr(world.link, "_link_flat", world.link.link)
    result = core(flat, _TiledRNG(world.stream7("sdl2-v11", "magnetic-epistemic"), L))
    levels = list(world.link.JGB_LEVELS)
    weights = np.array([world.link.JGB_POSTERIOR[j] for j in levels])
    jgb = float(world.stream7("sdl2-v11", "film").choice(levels, p=weights / weights.sum()))
    values = {q: np.asarray(result[q]["by_jgb"][jgb], float) for q in c7.LINK_OUTPUTS}
    if path == "c5":
        values["hci_kOe"] = state["hci_kOe"]
        values["br_kGs"] = state["br_kGs"] * rho / state["rho_stage_rel"]
    elif path != "link":
        raise ValueError("unknown magnetic hypothesis")
    bh = world.link._bhmax(values["br_kGs"], values["hci_kOe"], values["hk_over_hci"])
    over = values["hci_kOe"] > HCI_CEILING_KOE
    hci = np.where(over, HCI_CEILING_KOE, values["hci_kOe"])
    if over.any():
        bh = np.where(over, world.link._bhmax(values["br_kGs"], hci, values["hk_over_hci"]), bh)
    return {"hci_kOe": hci, "br_kGs": values["br_kGs"], "bhmax_MGOe": bh, "hk_over_hci": values["hk_over_hci"],
            "density_g_cm3": rho * world.rho_s, "raw_hci_kOe": values["hci_kOe"],
            "rho_made_rel": made["rho_body_rel"], "rho_stage_rel": state["rho_stage_rel"]}


def _made_fields(v, ox, rho_body):
    return {"rho_body_rel": rho_body, "W_as": v["W_bse"], "oxide_factor": ox, "ln_chem_offset": v["ln_chem_offset"],
            "grain_stage_um": {"as_sintered": v["g_as_sintered"], "after_900_520": v["g_after_900_520"]}}


def bodies_batch(world, route, *, path, lot_id, preparation_ids, specimen_ids, piece_id="P1"):
    """body_values for L bodies of one route at once, each with its own preparation and body effects (the package's
    per-draw arrays with the world's draw repeated). Equal to body_values body by body (checks/batch_parity.py)."""
    pm = world.pm
    prep, cond, steps = route_inputs(route)
    L = len(specimen_ids)
    controls = c7._manufacture_controls(cond, specimen_ids[0])
    executed = c7._preparation(prep, preparation_ids[0])
    prep_z = np.array([world.prep_z(lot_id, p) for p in preparation_ids])
    bodies = [world.body(s) for s in specimen_ids]
    u = np.array([b["u_spec"] for b in bodies])
    r_made = _tool_route(world, controls, executed, "none")
    mu, _ = pm._density_mean(pm._arrays(r_made), r_made, np.full(L, world.dmech), np.full(L, world.didx),
                             prep_z[:, None], "post")
    rho_body = np.clip(mu + np.array([b["rho_dev_g_cm3"] for b in bodies]) / world.rho_s, 0.3, 1.02)
    lat, v, ox = _latent(world, r_made, rho_body, u)
    made = _made_fields(v, ox, rho_body)
    key = c7.STAGE_ANNEAL[STAGES[len(steps)]]
    if key == "none":
        state = _stage(world, "none", controls["temperature_c"], lat, v, rho_body, None, v["lh"])
    else:
        hold_h = c7._first_hold_h(steps)
        r = _tool_route(world, controls, executed, key, hold_h)
        lat_s, v_s, _ = _latent(world, r, rho_body, u)
        state = _stage(world, key, r["T_peak_C"], lat_s, v_s, rho_body, np.full(L, hold_h),
                       _h4_lh(world, r, v_s["lh"], np.full(L, hold_h)))
    eta = np.array([world.piece_effect(s, piece_id) for s in specimen_ids])
    return _finish(world, c7.TOOL_STAGE[STAGES[len(steps)]], made, state, eta,
                   [(st, t, np.full(L, h)) for st, t, h in steps], path=path)


def median_table(world_config, routes):
    """Median body (every preparation, body and piece effect at zero) of each route, batched: route-level densities
    in one densification scan per feedstock (the package's route axis), one latent per sintering condition and anneal
    key, and the anneal layer, microstructure, link and bound over all routes at once. Equal route by route to
    body_values on MedianWorld (checks/batch_parity.py)."""
    base, path = world_config["base_world"], world_config["magnetic_path"]
    world = MedianWorld(base)
    pm = world.pm
    lot, pid, sid = MEDIAN
    sinter, items = {}, []
    for route in routes:
        prep, cond, steps = route_inputs(route)
        k = route_id(dict(route, anneal="none"))
        if k not in sinter:
            controls, executed = c7._manufacture_controls(cond, sid), c7._preparation(prep, pid)
            sinter[k] = {"controls": controls, "executed": executed,
                         "r": _tool_route(world, controls, executed, "none")}
        items.append((k, steps))
    # route-level density, one scan per feedstock over every sintering condition with it
    p = pm._params(pm.post, world.dmech, np.array([world.didx]))
    off = p["tau_prep"] * np.array([[0.0]])
    by_fd = {}
    for k, sc in sinter.items():
        by_fd.setdefault(sc["r"][world.tool.FEEDSTOCK_KEY], []).append(k)
    for keys in by_fd.values():
        As = [pm._arrays(sinter[k]["r"]) for k in keys]
        A = {"S": {f: np.concatenate([a["S"][f] for a in As], 0) for f in As[0]["S"]},
             **{f: np.concatenate([a[f] for a in As]) for f in As[0] if f != "S"}}
        mu = pm._dens_mu(p, world.dmech, A, off, sinter[keys[0]]["r"])[0]
        for k, m in zip(keys, mu):
            sinter[k]["rho_body"] = np.clip(np.array([m]), 0.3, 1.02)
    # latent per sintering condition (as made) and per anneal key
    zero = np.zeros(1)
    stage_groups = {}
    for i, (k, steps) in enumerate(items):
        sc = sinter[k]
        if "made" not in sc:
            lat, v, ox = _latent(world, sc["r"], sc["rho_body"], zero)
            sc["made"], sc["lat"] = _made_fields(v, ox, sc["rho_body"]), {"none": (lat, v)}
        key = c7.STAGE_ANNEAL[STAGES[len(steps)]]
        if key not in sc["lat"]:
            r = _tool_route(world, sc["controls"], sc["executed"], key, c7._first_hold_h(steps))
            lat, v, _ = _latent(world, r, sc["rho_body"], zero)
            sc["lat"][key] = (lat, v)
            sc.setdefault("r_key", {})[key] = r
        sc.setdefault("holds", {}).setdefault(key, []).append(i)
        stage_groups.setdefault((key, sc["r"]["T_peak_C"]), []).append(i)
    # C5's ln Hci at each route's state: correction 4, one call per sintering condition and anneal key over its holds
    lh_of = {}
    for k, sc in sinter.items():
        for key, idx in sc.get("holds", {}).items():
            v = sc["lat"][key][1]
            if key == "none":
                lh_of.update({i: v["lh"] for i in idx})
                continue
            lh = _h4_lh(world, sc["r_key"][key], np.repeat(v["lh"], len(idx)),
                        np.array([c7._first_hold_h(items[i][1]) for i in idx]))
            lh_of.update({i: lh[j:j + 1] for j, i in enumerate(idx)})
    # the anneal layer, batched over routes with the same anneal key and peak temperature
    cat = lambda xs: np.concatenate([np.asarray(x, float).reshape(-1) for x in xs])  # noqa: E731
    states = [None] * len(items)
    for (key, T), idx in stage_groups.items():
        lv = [sinter[items[i][0]]["lat"][key] for i in idx]
        lat = {f: cat([l[f] for l, _ in lv]) for f in ("g_as_sintered", "g_after_900_520")}
        v = {f: cat([x[f] for _, x in lv]) for f in ("lbr", "lh", "grain")}
        rho_body = cat([sinter[items[i][0]]["rho_body"] for i in idx])
        hold = None if key == "none" else np.array([c7._first_hold_h(items[i][1]) for i in idx])
        st = _stage(world, key, T, lat, v, rho_body, hold, cat([lh_of[i] for i in idx]))
        for j, i in enumerate(idx):
            states[i] = {f: x[j:j + 1] for f, x in st.items()}
    groups = {}
    for i, (k, steps) in enumerate(items):
        made = sinter[k]["made"]
        groups.setdefault((len(steps), made["oxide_factor"] is None), []).append((i, made, states[i], steps))
    out = [None] * len(items)
    for (n_steps, _), members in groups.items():   # one stage, and the oxide term present for all or none
        made = {"rho_body_rel": cat([m["rho_body_rel"] for _, m, _, _ in members]),
                "W_as": cat([m["W_as"] for _, m, _, _ in members]),
                "oxide_factor": None if members[0][1]["oxide_factor"] is None else
                cat([m["oxide_factor"] for _, m, _, _ in members]),
                "ln_chem_offset": cat([m["ln_chem_offset"] for _, m, _, _ in members]),
                "grain_stage_um": {s: cat([m["grain_stage_um"][s] for _, m, _, _ in members])
                                   for s in ("as_sintered", "after_900_520")}}
        state = {f: cat([st[f] for _, _, st, _ in members]) for f in ("rho_stage_rel", "hci_kOe", "br_kGs", "grain_um")}
        steps = [(st, t, np.array([s[j][2] for _, _, _, s in members])) for j, (st, t, _) in enumerate(members[0][3])]
        vals = _finish(world, c7.TOOL_STAGE[STAGES[n_steps]], made, state, np.zeros(len(members)), steps, path=path)
        for j, (i, _, _, _) in enumerate(members):
            out[i] = {f: float(x[j]) for f, x in vals.items()}
    return out


def meets(values, margin=1.0):
    return (values["hci_kOe"] >= margin * REQUIREMENT["hci_kOe"]
            and values["bhmax_MGOe"] >= margin * REQUIREMENT["bhmax_MGOe"])


def owner_salt(world_config):
    """Confirmation-lot identities no agent can guess (derived from the hidden world record)."""
    return hashlib.sha256(c7._canonical(world_config)).hexdigest()[:20]


def route_p_all(world, route, *, path, salt, lots=64, start=0):
    """P(all three bodies of one fresh preparation meet), from `lots` fresh confirmation lots (new preparation and
    bodies, owner-only identities), all bodies in one batch. Returns (passing lots, lots). The lot identities do not
    depend on the route (common random numbers): every route meets the same confirmation lots, so two routes that make
    the same bodies in a world always get the same verdict."""
    preps = [f"OWNER-{salt}-L{k}" for k in range(start, start + lots)]
    n = REQUIREMENT["bodies"]
    v = bodies_batch(world, route, path=path, lot_id="OWNER-CONFIRM", preparation_ids=[p for p in preps for _ in range(n)],
                     specimen_ids=[f"{p}-B{b}" for p in preps for b in range(n)])
    ok = (v["hci_kOe"] >= REQUIREMENT["hci_kOe"]) & (v["bhmax_MGOe"] >= REQUIREMENT["bhmax_MGOe"])
    return int(ok.reshape(lots, n).all(axis=1).sum()), lots


def sequential_p_all(world, route, *, path, salt, batch=LOT_BATCH, min_lots=MIN_LOTS, max_lots=MAX_LOTS):
    """P(all 3) from fresh lots (the same owner lot identities every time): the first min_lots at once, then batches,
    stopping once a 95% Wilson interval excludes the 0.5 standard. The one estimator for both world feasibility and a
    nominated route's outcome, so the two can never disagree."""
    k, n = route_p_all(world, route, path=path, salt=salt, lots=min_lots, start=0)
    while True:
        p, z = k / n, 1.96
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        if centre - half > REQUIREMENT["p_all"] or centre + half < REQUIREMENT["p_all"] or n >= max_lots:
            return k, n
        dk, dn = route_p_all(world, route, path=path, salt=salt, lots=batch, start=n)
        k, n = k + dk, n + dn


def world_truth(world_config, *, screen_margin=None, max_lots=MAX_LOTS):
    """Family truth for a v1.2 world: every declared route's median body, and P(all 3) where it can matter.
    Routes whose zero-effect body misses the requirement by more than 3% are not sampled (p_all recorded as 0): P(all 3)
    >= 0.5 needs P(one body meets) >= 0.5, which a body distribution centred below the bar does not give."""
    screen_margin = SCREEN_MARGIN if screen_margin is None else screen_margin
    base, path = world_config["base_world"], world_config["magnetic_path"]
    world = c7.World(base)
    salt = owner_salt(world_config)
    routes = declared_routes()
    rows, max_raw_hci = [], 0.0
    for route, med in zip(routes, median_table(world_config, routes)):
        max_raw_hci = max(max_raw_hci, med["raw_hci_kOe"])
        row = {"route": route, "id": route_id(route), "median": med, "p_all": 0.0, "lots": 0,
               "lps_extrapolated": bool(route["anneal"] != "none" and world.amech == "LPS"
                                        and med["rho_made_rel"] < LPS_START_FLOOR_REL)}
        if meets(med, screen_margin):
            k, n = sequential_p_all(world, route, path=path, salt=salt, max_lots=max_lots)
            row.update(p_all=k / n, lots=n)
        rows.append(row)
    passing = [r for r in rows if r["p_all"] >= REQUIREMENT["p_all"]]
    best = max(rows, key=lambda r: (r["p_all"], min(r["median"]["hci_kOe"] / REQUIREMENT["hci_kOe"],
                                                    r["median"]["bhmax_MGOe"] / REQUIREMENT["bhmax_MGOe"])))
    return {"version": VERSION, "feasible": bool(passing), "passing": [r["id"] for r in passing],
            "best": best["id"], "best_p_all": best["p_all"], "routes": rows, "max_raw_hci_kOe": max_raw_hci,
            "magnetic_path": path, "draw": base["draw"],
            "lps_only": bool(passing) and all(r["lps_extrapolated"] for r in passing)}


def joint_ratio(hci_kOe, bhmax_MGOe):
    """How close a body is to the requirement: the smaller of its two ratios to the bars (1 = meets both)."""
    return min(hci_kOe / REQUIREMENT["hci_kOe"], bhmax_MGOe / REQUIREMENT["bhmax_MGOe"])


def best_route_set(ratios):
    """Routes whose median body is as close to the requirement as the world's best, within a tolerance that grows with
    the remaining shortfall: max(0.05, half of 1 - best). Measuring one of these is what a restriction needs."""
    best = max(ratios.values())
    tolerance = max(0.05, 0.5 * (1.0 - best))
    return {rid for rid, x in ratios.items() if x >= best - tolerance}, best, tolerance


def route_outcome(world_config, route):
    """Rung-1 truth for one nominated route: P(all 3 bodies of a fresh preparation meet) >= 0.5, by the same
    sequential estimator and owner lots world_truth uses (its screen aside: a route below the screen fails there)."""
    route = canonical_route(route)
    base, path = world_config["base_world"], world_config["magnetic_path"]
    med = median_table(world_config, [route])[0]
    if not meets(med, SCREEN_MARGIN):
        return {"route": route_id(route), "p_all": 0.0, "lots": 0, "meets": False, "screened": True}
    k, n = sequential_p_all(c7.World(base), route, path=path, salt=owner_salt(world_config))
    return {"route": route_id(route), "p_all": k / n, "lots": n, "meets": k / n >= REQUIREMENT["p_all"]}


def draw(package_dir, world_seed, world_id, measurement, *, lot_id, drawn_at=None, path=None):
    """One v1.2 world: a C7.3 draw plus the magnetic-path structural draw (equal odds unless fixed)."""
    base = c7.draw_world(package_dir, world_seed, world_id, measurement, drawn_at=drawn_at, lot_id=lot_id)
    if path is None:
        rng = np.random.default_rng([int(world_seed), 1202])
        path = ("link", "c5")[int(rng.integers(0, 2))]
    return {"family": FAMILY, "version": VERSION, "base_world": base, "magnetic_path": path, "magnetic_log_sd": .02,
            "requirement": copy.deepcopy(REQUIREMENT)}
