"""SDL-2 v1.1 development adapter over the released C7 campaign implementation.

Owner only. Adds persistent hypothetical magnetic truth, not physical calibration.
All process, density, piece evolution and nonmagnetic instruments remain C7's.
The live C7 family and its model-distribution interface are unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from . import c7_campaign as c7

FAMILY = "ndfeb_sdl2_v11_development"
VERSION = "sdl2-v1.1-dev.1"
CORRECTED_VERSION = "sdl2-v1.1-dev.2"


def magnetic_state(world, state, sid, piece, steps, *, path="link", normalization=None):
    """Common world-level epistemic draws across stages; persistent body/piece effects.

    The same seed is used for the link's epistemic quantities at each state. This
    is common-random-number coupling, not independent posterior resampling.
    The density in the link evolves from the piece's initial local porosity.
    C5 is a second structural hypothesis; its Br is the existing C7 C5 path.
    Neither hypothesis is experimentally validated on the current panel.
    """
    eta = world.piece_effect(sid, piece)
    rho = float(world.an.piece_density(
        world.an_p, world.an_mech, np.array([state["rho_body_rel"]]),
        np.array([eta]), [(t, h / 3600.0) for _, t, h in steps])[0])
    density = rho * world.rho_s
    flat = {k: np.asarray(v)[:1] for k, v in
            world.link_flat_state(state["c6"], state["tool_stage"], rho).items()}
    rng = world.stream7("sdl2-v11", "magnetic-epistemic")
    core = getattr(world.link, "_link_flat", world.link.link)
    result = core(flat, rng)
    levels = list(world.link.JGB_LEVELS)
    weights = np.array([world.link.JGB_POSTERIOR[j] for j in levels])
    jgb = float(world.stream7("sdl2-v11", "film").choice(levels, p=weights / weights.sum()))
    values = {q: float(result[q]["by_jgb"][jgb][0]) for q in c7.LINK_OUTPUTS}
    if path == "c5":
        values["hci_kOe"] = state["hci_kOe"]
        values["br_kGs"] = state["br_kGs"] * rho / state["rho_stage_rel"]
        # Curve shape is explicitly the shared link shape for this adapter's
        # C5-Hci/Br structural alternative, not a full C5 likelihood replay.
    elif path != "link":
        raise ValueError("unknown magnetic hypothesis")
    if normalization is not None:
        values["br_kGs"] *= float(normalization) / density
    values["bhmax_MGOe"] = float(world.link._bhmax(
        np.array([values["br_kGs"]]), np.array([values["hci_kOe"]]),
        np.array([values["hk_over_hci"]]))[0])
    return values, density


def request_states(request, world):
    """Recover states from the exact canonical request used by native execution."""
    preparations = {p["preparation_id"]: c7._preparation(p["conditions"], p["preparation_id"])
                    for p in request.get("preparations", [])}
    for specimen in request["specimens"]:
        sid, pid = specimen["specimen_id"], specimen["preparation_id"]
        condition = specimen["conditions"]
        context, piece = c7._context(condition, sid)
        if request.get("operation", "manufacture") == "manufacture":
            controls, executed, steps, stage = c7._manufacture_controls(condition, sid), preparations[pid], [], "as_sintered"
        else:
            controls, executed, steps, stage = c7._history(specimen["prior_processing_conditions"], sid)
            if request["operation"] == "anneal":
                extra = c7._anneal_controls(condition, sid, stage, request["output_stage"])
                steps += [(s["stage"], float(s["temperature_c"]), float(s["hold_time_s"])) for s in extra]
                stage = request["output_stage"]
        state = world.body_state(request["lot_id"], pid, sid, controls, executed, stage, c7._first_hold_h(steps))
        yield sid, context, piece, steps, state


def simulate_sdl2_v11(request, *, world_config, seed):
    if world_config.get("interface_version") == CORRECTED_VERSION:
        return _simulate_v2(request, world_config=world_config, seed=seed)
    if world_config["family"] != FAMILY:
        raise ValueError("wrong SDL-2 family")
    config = world_config["base_world"]
    world = c7.World(config)
    # Existing validation, specimen history, density and metadata unchanged.
    delivery = c7.simulate_c7_campaign(request, world_config=config, seed=seed)
    sigma = float(world_config.get("magnetic_log_sd", 0.02))
    if not math.isfinite(sigma) or not 0 < sigma <= 0.2:
        raise ValueError("magnetic_log_sd must be finite in (0, .2]")
    states = {sid: (context, piece, steps, state) for sid, context, piece, steps, state in request_states(request, world)}
    rhash = hashlib.sha256(c7._canonical(request)).hexdigest()
    for observation in delivery["observations"]:
        sid, q = observation["specimen_id"], observation["quantity"]
        if q not in c7.MAGNETIC:
            continue
        context, piece, steps, state = states[sid]
        norm = context.get("normalization_density_g_cm3")
        true, density = magnetic_state(world, state, sid, piece, steps,
                                      path=world_config.get("magnetic_path", "link"), normalization=norm)
        # One loop shares readings across Hci/Br/BHmax; re-requesting the exact
        # canonical request is idempotent. A different acquisition has new noise.
        rng = world.stream7("sdl2-v11", "loop-reading", sid, piece, state["tool_stage"], rhash)
        read = dict(true)
        for key in ("hci_kOe", "br_kGs"):
            read[key] *= math.exp(sigma * float(rng.standard_normal()))
        read["bhmax_MGOe"] = float(world.link._bhmax(
            np.array([read["br_kGs"]]), np.array([read["hci_kOe"]]),
            np.array([read["hk_over_hci"]]))[0])
        observation.pop("model_distribution", None)
        observation.update(status="conditional_synthetic", method="sdl2_v11_hypothetical_corrected_loop",
                           value=(read if q == "demagnetization_curve" else read[c7.LINK_SOURCE[q]]),
                           piece_id=piece, measurement_temperature_C=20.0,
                           orientation="isotropic", loop_id=f"{rhash[:16]}:{sid}:{piece}",
                           measurement_statistic="synthetic_loop_not_physical_measurement",
                           uncertainty={"log_sd_hci": sigma, "log_sd_br": sigma,
                                        "curve_shape": "fixed scenario value; not independently measured"},
                           normalization={"density_g_cm3": density if norm is None else float(norm),
                                          "basis": "own_piece_stage" if norm is None else "stated_density"},
                           reason="Hypothetical persistent specimen; no current-panel magnetic validation",
                           limits=["Room temperature; isotropic; model-based curve; no operating-temperature verification",
                                   "Independent log reading errors for Hci and Br; BHmax derived from the same curve",
                                   "Two per cent error is a declared scenario, not qualified RPI measurement precision"])
        # The coherent curve parameters are observations, not hidden truth.
        observation["loop_reading"] = (read if q in ("demagnetization_curve", "bhmax_MGOe")
                                        else {c7.LINK_SOURCE[q]: read[c7.LINK_SOURCE[q]]})
    delivery["provenance"].update(evaluation_version=VERSION, magnetic_model_status="hypothetical",
                                  physical_execution=False,
                                  request_world_seed_sha256=hashlib.sha256(c7._canonical(
                                      {"request": request, "world": world_config, "seed": seed})).hexdigest())
    return delivery


def _simulate_v2(request, *, world_config, seed):
    """Versioned density/loop acquisition correction; C7 process physics unchanged.

    The accepted field_set_id is the explicit acquisition identifier for these
    two instruments. Quantity selection is a view of that acquisition, not a
    source of independent reading noise. Other C7 instruments keep their existing
    sampling contract. Ideal magnetic volume normalization is a declared world
    convention, not a measurement of the hidden piece density.
    """
    corrected = {"density_g_cm3", *c7.MAGNETIC}
    for spec in request["specimens"]:
        if set(spec.get("measurements", request.get("measurements", []))) & corrected:
            context = spec["conditions"].get("measurement_context", {})
            acquisition = context.get("field_set_id")
            if not isinstance(acquisition, str) or not acquisition.strip():
                raise ValueError("density/loop acquisition requires measurement_context.field_set_id; "
                                 "retry the unchanged request for redelivery, or choose a new ID for a fresh reading")
            if "normalization_density_g_cm3" in context:
                raise ValueError("this evaluation uses declared ideal own-piece magnetic normalization; "
                                 "user-specified density rescaling is not modeled; request density separately")
    legacy = dict(world_config)
    legacy.pop("interface_version")
    delivery = simulate_sdl2_v11(request, world_config=legacy, seed=seed)
    world = c7.World(world_config["base_world"])
    states = {sid: (context, piece, steps, state) for sid, context, piece, steps, state in request_states(request, world)}
    world_key = hashlib.sha256(c7._canonical(world_config)).hexdigest()
    specs = {s["specimen_id"]: s for s in request["specimens"]}
    sigma = float(world_config.get("magnetic_log_sd", .02))
    for row in delivery["observations"]:
        sid, quantity = row["specimen_id"], row["quantity"]
        if quantity not in corrected:
            continue
        context, piece, steps, state = states[sid]
        acquisition = context["field_set_id"]
        identity = {"world": world_key, "lot": request["lot_id"], "specimen": sid,
                    "preparation": specs[sid]["preparation_id"], "cycle": specs[sid]["cycle_id"],
                    "piece": piece, "stage": state["tool_stage"], "acquisition": acquisition}
        key = hashlib.sha256(c7._canonical(identity)).hexdigest()
        row.update(acquisition_id=acquisition, evaluation_version=CORRECTED_VERSION)
        true, density = magnetic_state(world, state, sid, piece, steps,
                                      path=world_config.get("magnetic_path", "link"))
        if quantity == "density_g_cm3":
            method = context.get("density_method", "archimedes")
            rho = np.array([density / world.rho_s])
            mean = float(world.an.reading_mean(world.an_p, rho, method)[0])
            sd = float(world.an.reading_sd(world.an_p, method)[0])
            noise = float(world.stream7(CORRECTED_VERSION, "density-reading", key, method).standard_normal())
            row["value"] = (mean + sd * noise) * world.rho_s
            row["observation_id"] = f"{key[:16]}:{sid}:{piece}:{method}:{quantity}"
        else:
            rng = world.stream7(CORRECTED_VERSION, "loop-reading", key)
            reading = dict(true)
            for q in ("hci_kOe", "br_kGs"):
                reading[q] *= math.exp(sigma * float(rng.standard_normal()))
            reading["bhmax_MGOe"] = float(world.link._bhmax(
                np.array([reading["br_kGs"]]), np.array([reading["hci_kOe"]]),
                np.array([reading["hk_over_hci"]]))[0])
            row.update(value=reading if quantity == "demagnetization_curve" else reading[c7.LINK_SOURCE[quantity]],
                       loop_id=f"{key[:16]}:{sid}:{piece}",
                       observation_id=f"{key[:16]}:{sid}:{piece}:{quantity}",
                       normalization={"basis": "ideal_own_piece_stage", "status": "declared_simulation_assumption"})
            row["loop_reading"] = (reading if quantity in ("demagnetization_curve", "bhmax_MGOe")
                                   else {c7.LINK_SOURCE[quantity]: reading[c7.LINK_SOURCE[quantity]]})
            row["limits"] += ["Ideal own-piece volume normalization is assumed, not measured; no numerical density is disclosed and physical normalization uncertainty is not modeled"]
    delivery["provenance"].update(evaluation_version=CORRECTED_VERSION,
                                  request_world_seed_sha256=hashlib.sha256(c7._canonical(
                                      {"request": request, "world": world_config, "seed": seed})).hexdigest())
    return delivery


def truth_for(request, world_config):
    """Owner-only values for the precise requests in the executed ledger."""
    world = c7.World(world_config["base_world"])
    rows = {}
    for sid, context, piece, steps, state in request_states(request, world):
        values, density = magnetic_state(world, state, sid, piece, steps,
                                         path=world_config.get("magnetic_path", "link"))
        key = (sid, piece, state["tool_stage"])
        rows[key] = {**values, "density_g_cm3": density}
    return rows
