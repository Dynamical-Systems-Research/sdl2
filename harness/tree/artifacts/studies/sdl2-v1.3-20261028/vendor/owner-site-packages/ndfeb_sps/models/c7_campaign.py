"""Owner-side campaign world (family ``ndfeb_c7_campaign_v1``): one hidden draw from a named C7 package.

C7 (2026-10-23) extends family ``ndfeb_c6_campaign_v1`` (whose text follows) with the package's anneal layer: each world
also holds one anneal draw (a densification mechanism and one of its stored draws, by the package's weights). A body's
density at an annealed stage is its as-made density carried through the executed anneal; the stage density feeds the
density reading, the fracture-intercept feature factor, polished-section planarity and porosity, and the coercivity
link's Br. Density readings name their method (archimedes or geometric) and piece; the same piece at two stages shares
its piece effect. The 900 C hold is a control (1800-14400 s), the cooling mode is recorded (no modeled effect), a
polished-section request on a fracture surface returns a refusal, and magnetic readings can carry a stated
normalization density. New lot and family names: no C6 body can be continued here. C5 and C6 quantities keep their
streams and measurement models except where this paragraph names a change.

A world is a single posterior draw of the owner-built physics tool C6 (Dynamical's model fitted to RPI's released
records, with its microstructure layer): the C5 draw (a density mechanism and one of its stored draws; a coercivity
mechanism, a load-timing explanation and one of their stored draws, chosen with the package's own weights and
draw-assignment functions, exactly as family ``ndfeb_c5_campaign_v1`` draws them), the package's declared C5 layers at
those indices, the C6 declared draws (c6_*) of the same coercivity index, and a per-world seed. The boundary film's
magnetization (J_GB) is never drawn and never stored: the coercivity link marginalizes it inside every call.

Built from ``c5_campaign.py``. The C5 quantities (density_g_cm3, grain_size_nm, the BSE class reading) use the C5
family's measurement models and its deterministic streams (namespace ``ndfeb_c5_campaign_v1``), so a C6 world with the
same seed, draw and lot returns exactly the C5 family's values for them. New quantities use their own streams
(namespace ``ndfeb_c6_campaign_v1``). Per-draw values come from the package's own functions (the tool's route, density,
degassing, magnetic-latent and oxide-factor functions, microstructure.py and coercivity_link.py) evaluated at the
world's indices. This module is an owner-side native provider dependency, never an investigator answer key, and it
reads no RPI result beyond what the package contains.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

from .conditional_workflow import _resolved_anneal_steps, validate_anneal_condition

FAMILY = "ndfeb_c7_campaign_v1"
MODEL_VERSION = "ndfeb-c7-campaign-v1"
PREVIEW_DOMAIN = "c7-campaign-v1"   # preview_task.anneal_model_domain that selects this family's control check
LOT_ID = "CAMPAIGN-C7-LOT"   # the C7 world's lot; a later package's world names its own (draw_world lot_id), so a body
                             # made under one package can never be continued under another
C5_STREAMS = "ndfeb_c5_campaign_v1"  # stream namespace of the C5 quantities (identical to the C5 family's streams)
PACKAGE_FILES = {"tool": "ndfeb_physics_tool.py", "physics": "physics.py", "forward": "forward.py",
                 "microstructure": "microstructure.py", "coercivity_link": "coercivity_link.py",
                 "anneal": "anneal.py", "anneal_posterior": "anneal_posterior.npz",
                 "posterior": "posterior_tool.npz", "manifest": "TOOL_MANIFEST.json"}
C7_STREAMS = "ndfeb_c7_campaign_v1"  # stream namespace of the C7 quantities (density readings, piece effects)
OPTIONAL_PACKAGE_FILES = frozenset({"manifest"})   # absent only for an owner development snapshot (unverified)
WORLD_KEYS = frozenset({"family", "world_id", "package", "draw", "world_seed", "lots", "measurement",
                        "declared_layer", "provenance"})
MEASUREMENT_KEYS = ("density_body_sd_g_cm3", "link_model_draws")   # C7: reading sd comes from the anneal draw
LINK_DRAWS_RANGE = (500, 20000)
COOLING_RATE_C_MIN = 100.0   # the tool's default SPS cooling rate (README "Routes"); converts pressure_full_at_c
SUPPORT_ENVELOPE = {
    "temperature_c": (650.0, 950.0), "heating_rate_c_min": (10.0, 100.0), "hold_time_s": (60.0, 1800.0),
    "pressure_mpa": (30.0, 60.0), "pressure_full_at_c": (600.0, 950.0), "mass_g": (0.01, 25.0),
    "degassing_temperature_c": (550.0, 700.0), "degassing_hold_time_s": (0.0, 3600.0),
    "degassing_completeness": (0.0, 1.5), "feedstock_d50_um": (0.5, 8.0), "coarse_threshold_nm": (1.0, 100000.0),
    "first_anneal_hold_time_s": (1800.0, 14400.0), "normalization_density_g_cm3": (4.0, 8.0),
    "status": "declared conditional support envelope (the span of the tool's fitted records for temperature); "
              "outside it no response is claimed"}
FITTED_DEGAS = {"temperature_c": 600.0, "hold_time_s": (600.0, 900.0)}  # the tool's fitted D/E state (completeness 1)
SCHEDULES = {"on_heating": "heating", "on_cooling": "cooling", "on_dwell": "dwell"}
DEGAS_METHODS = ("vacuum_degassing", "none")
CURRENT_PATHS = ("compact", "insulated")
STAGE_ANNEAL = {"as_sintered": "none", "after_first_anneal": "900", "after_second_anneal": "900+520"}
TOOL_STAGE = {"as_sintered": "as_sintered", "after_first_anneal": "after_900", "after_second_anneal": "after_900_520"}
C5_ANNEAL_STEPS = {"after_first_anneal": (900.0, 7200.0), "after_second_anneal": (520.0, 7200.0)}
ANNEAL_TEMPERATURES = {"after_first_anneal": 900.0, "after_second_anneal": 520.0}
SECOND_ANNEAL_HOLD_S = 7200.0
COOLING_MODES = ("furnace_cooling", "argon_quench")   # recorded; no modeled effect in this world (declared)
DENSITY_METHODS = ("archimedes", "archimedes_saturated", "geometric")   # saturated: package anneal layer >= C7.1
SURFACES = ("polished", "fracture")
VOCABULARY = ("bhmax_MGOe", "boundary_chemistry", "br_kGs", "coarse_number_fraction", "demagnetization_curve",
              "density_g_cm3", "grain_d90_nm", "grain_size_nm", "hci_kOe", "hci_screen_kOe", "microstructure",
              "porosity_area_fraction", "powder_icp_oes", "sps_log")
UNITS = {"density_g_cm3": "g/cm^3", "grain_size_nm": "nm", "grain_d90_nm": "nm", "coarse_number_fraction": "1",
         "porosity_area_fraction": "1", "hci_kOe": "kOe", "br_kGs": "kGs", "bhmax_MGOe": "MGOe",
         "hci_screen_kOe": "kOe"}
MAGNETIC = ("hci_kOe", "br_kGs", "bhmax_MGOe", "hci_screen_kOe", "demagnetization_curve")
LINK_OUTPUTS = ("hci_kOe", "br_kGs", "hk_over_hci", "bhmax_MGOe")
LINK_SOURCE = {"hci_kOe": "hci_kOe", "br_kGs": "br_kGs", "bhmax_MGOe": "bhmax_MGOe", "hci_screen_kOe": "hci_kOe"}
SECTION_KEYS = ("porosity_area_fraction", "microstructure", "boundary_chemistry")
# The tool's REFUSALS constant (ndfeb_physics_tool.REFUSALS), copied so world-free validation (the contract preview)
# reads no package; at delivery the package's own constant is used, and a test asserts the two are equal.
REFUSALS = {"powder_icp_oes": "no model of powder chemistry by ICP-OES",
            "film_magnetization": "the 2-3 nm boundary film's magnetization is not observable by any measurement in "
                                  "this window (STEM-EDS or atom-probe tomography can constrain chemistry, not directly measure film magnetization)",
            "stem_eds": "no STEM or atom-probe instrument is simulated in this window"}
REFUSED_IN_VOCABULARY = ("powder_icp_oes",)
FILM_NAMES = frozenset({"film_magnetization", "boundary_film_magnetization", "gb_film_magnetization", "j_gb", "jgb",
                        "film_magnetisation", "grain_boundary_magnetization", "boundary_magnetization"})
STEM_NAMES = frozenset({"stem_eds", "stem", "tem", "stem_haadf", "haadf_stem", "eds_stem", "atom_probe", "apt",
                        "atom_probe_tomography", "tem_eds", "eels"})
BSE_TEXT = {"spread": "Nd-rich phase spread along grain boundaries",
            "pooled": "Nd-rich phase pooled at triple junctions"}
FILM_STATEMENT = "film magnetization: not observable here"
SPS_DIE_MM = 10.0
SPS_DT_S = 1.0
CONTEXT_KEYS = frozenset({"field_count", "features_per_field", "field_set_id", "selection", "coarse_threshold_nm",
                          "piece_id", "density_method", "surface", "normalization_density_g_cm3"})
MANUFACTURE_KEYS = frozenset({"temperature_c", "heating_rate_c_min", "hold_time_s", "pressure_mpa",
                              "pressure_schedule", "pressure_full_at_c", "current_path", "mass_g", "diameter_mm",
                              "measurement_context", "route", "operation", "output_stage"})
MEASURE_KEYS = frozenset({"mass_g", "measurement_context", "operation", "output_stage", "input_stage",
                          "process_stage"})
ANNEAL_KEYS = frozenset({"anneal_steps", "mass_g", "measurement_context", "operation", "output_stage",
                         "input_stage", "process_stage", "cooling_mode"})
PREPARATION_KEYS = frozenset({"method", "temperature_c", "hold_time_s", "degassing_completeness",
                              "feedstock_d50_um"})
EVIDENCE_SCOPE = "conditional_computation_not_RPI_measurement"
LIMIT = "Conditional computation from a hidden draw of the owner-built physics tool C7; not a physical measurement"
MAGNETIC_STATUS = "simulated_model_distribution"
MAGNETIC_REASON = ("model distribution, not a measurement: the coercivity link's distribution at this body's hidden "
                   "microstructure, marginalized over the boundary film's magnetization, which no world holds and no "
                   "measurement in this window can reveal; simulated and unscored in this window, so value is null")


# ------------------------------------------------------------------ small helpers

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _number(value, name, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} {value:g} outside this world's declared support envelope [{low:g}, {high:g}]")
    return float(value)


def _context(condition, sid):
    context = condition.get("measurement_context", {})
    if not isinstance(context, dict) or set(context) - CONTEXT_KEYS:
        raise ValueError(f"{sid}: measurement_context only accepts {sorted(CONTEXT_KEYS)}")
    piece = context.get("piece_id", "1")
    if not isinstance(piece, str) or not piece.strip() or len(piece) > 64:
        raise ValueError(f"{sid}: measurement_context.piece_id must be a nonempty identifier")
    if context.get("density_method", "archimedes") not in DENSITY_METHODS:
        raise ValueError(f"{sid}: measurement_context.density_method must be one of {list(DENSITY_METHODS)}")
    if context.get("surface", "polished") not in SURFACES:
        raise ValueError(f"{sid}: measurement_context.surface must be one of {list(SURFACES)} (SEM grain and fracture "
                         "readings are always fracture-surface readings; this selects the surface of a section reading)")
    if "normalization_density_g_cm3" in context:
        low, high = SUPPORT_ENVELOPE["normalization_density_g_cm3"]
        _number(context["normalization_density_g_cm3"], f"{sid}: measurement_context.normalization_density_g_cm3",
                low, high)
    return context, piece


def _refusal(name):
    """Explicit reason for a measurement name outside the vocabulary that asks for the film or a STEM readout."""
    key = str(name).strip().lower()
    if key in FILM_NAMES or ("magneti" in key and ("film" in key or "boundary" in key)):
        return f"{name}: not offered in this world: {REFUSALS['film_magnetization']}"
    if key in STEM_NAMES or key.startswith(("stem_", "tem_", "atom_probe")):
        return f"{name}: not offered in this world: {REFUSALS['stem_eds']}"
    return None


# ------------------------------------------------------------------ world-free control validation

def _preparation(conditions, where):
    """Executed powder batch: degassing occurrence and completeness, feedstock size (one batch, one value each)."""
    if not isinstance(conditions, dict):
        raise ValueError(f"{where}: preparation requires a conditions object")
    unknown = set(conditions) - PREPARATION_KEYS
    if unknown:
        raise ValueError(f"{where}: unsupported preparation controls {sorted(unknown)}; use {sorted(PREPARATION_KEYS)}")
    method = conditions.get("method")
    if method not in DEGAS_METHODS:
        raise ValueError(f"{where}: preparation method must be one of {list(DEGAS_METHODS)}")
    low, high = SUPPORT_ENVELOPE["feedstock_d50_um"]
    d50 = _number(conditions.get("feedstock_d50_um", 4.0), f"{where}: feedstock_d50_um", low, high)
    if method == "none":
        if ("temperature_c" in conditions or conditions.get("hold_time_s", 0) != 0
                or "degassing_completeness" in conditions):
            raise ValueError(f"{where}: method none declares no degassing temperature, hold or completeness")
        return {"method": "none", "temperature_c": None, "hold_time_s": 0.0, "feedstock_d50_um": d50,
                "degassing_completeness": None}
    low, high = SUPPORT_ENVELOPE["degassing_temperature_c"]
    temperature = _number(conditions.get("temperature_c"), f"{where}: degassing temperature_c", low, high)
    low, high = SUPPORT_ENVELOPE["degassing_hold_time_s"]
    hold = _number(conditions.get("hold_time_s"), f"{where}: degassing hold_time_s", low, high)
    if hold <= 0:
        raise ValueError(f"{where}: vacuum_degassing requires a positive hold_time_s; use method none for no degassing")
    completeness = conditions.get("degassing_completeness")
    if completeness is None:
        lo, hi = FITTED_DEGAS["hold_time_s"]
        if temperature != FITTED_DEGAS["temperature_c"] or not lo <= hold <= hi:
            raise ValueError(
                f"{where}: this world has no degassing-kinetics model; at set-points other than 600 C for 600-900 s, "
                "state degassing_completeness (a number in [0, 1.5], 1 = the tool's fitted degassed state, or "
                "'complete')")
        completeness = 1.0
    elif completeness != "complete":
        low, high = SUPPORT_ENVELOPE["degassing_completeness"]
        completeness = _number(completeness, f"{where}: degassing_completeness", low, high)
    return {"method": "vacuum_degassing", "temperature_c": temperature, "hold_time_s": hold, "feedstock_d50_um": d50,
            "degassing_completeness": completeness}


def _manufacture_controls(condition, sid):
    unknown = set(condition) - MANUFACTURE_KEYS
    if unknown:
        raise ValueError(f"{sid}: unsupported manufacture controls {sorted(unknown)}; "
                         f"use {sorted(MANUFACTURE_KEYS - {'operation', 'output_stage'})}")
    controls = {}
    for key in ("temperature_c", "heating_rate_c_min", "hold_time_s", "pressure_mpa"):
        if key not in condition:
            raise ValueError(f"{sid}: manufacture requires explicit {key}")
        low, high = SUPPORT_ENVELOPE[key]
        controls[key] = _number(condition[key], f"{sid}: {key}", low, high)
    schedule = condition.get("pressure_schedule")
    if schedule not in SCHEDULES:
        raise ValueError(f"{sid}: pressure_schedule must be one of {list(SCHEDULES)}")
    controls["pressure_schedule"] = schedule
    if schedule == "on_cooling":
        if condition.get("pressure_full_at_c") is not None:
            low, high = SUPPORT_ENVELOPE["pressure_full_at_c"]
            full = _number(condition["pressure_full_at_c"], f"{sid}: pressure_full_at_c", low, high)
            if full > controls["temperature_c"]:
                raise ValueError(f"{sid}: pressure_full_at_c must not exceed the peak temperature_c")
            controls["pressure_full_at_c"] = full
    elif condition.get("pressure_full_at_c") is not None:
        raise ValueError(f"{sid}: pressure_full_at_c applies only to pressure_schedule on_cooling")
    path = condition.get("current_path", "compact")
    if path not in CURRENT_PATHS:
        raise ValueError(f"{sid}: current_path must be one of {list(CURRENT_PATHS)}")
    controls["current_path"] = path
    low, high = SUPPORT_ENVELOPE["mass_g"]
    controls["mass_g"] = _number(condition.get("mass_g"), f"{sid}: mass_g", low, high)
    if "diameter_mm" in condition:
        controls["diameter_mm"] = _number(condition["diameter_mm"], f"{sid}: diameter_mm", 1, 50)
    return controls


def _anneal_controls(condition, sid, input_stage, output_stage):
    extra = set(condition) - ANNEAL_KEYS
    if extra:
        raise ValueError(f"{sid}: anneal controls belong in canonical anneal_steps; unsupported {sorted(extra)}")
    if "anneal_steps" not in condition:
        raise ValueError(f"{sid}: submit explicit canonical anneal_steps (stage, temperature_c, hold_time_s, "
                         "heating_rate_c_min for each step)")
    validate_anneal_condition(condition, input_stage=input_stage, output_stage=output_stage)
    if condition.get("cooling_mode", "furnace_cooling") not in COOLING_MODES:
        raise ValueError(f"{sid}: cooling_mode must be one of {list(COOLING_MODES)}")
    steps = _resolved_anneal_steps(condition, {"anneal_heating_rate_c_min": 10.0}, input_stage, output_stage, 5.0)
    low, high = SUPPORT_ENVELOPE["first_anneal_hold_time_s"]
    for step in steps:
        T, hold = float(step["temperature_c"]), float(step["hold_time_s"])
        if step["stage"] == "after_first_anneal" and not (T == 900.0 and low <= hold <= high):
            raise ValueError(
                f"{sid}: not supported by this world: C7 models the first anneal at 900 C for {low:g}-{high:g} s "
                f"(densification calibrated on #8's 900 C return; magnetic anneal terms are historical E's 2 h); got "
                f"{T:g} C for {hold:g} s. This is a limit of this instrument, not of the process")
        if step["stage"] == "after_second_anneal" and (T, hold) != (520.0, SECOND_ANNEAL_HOLD_S):
            raise ValueError(
                f"{sid}: not supported by this world: C7 models the second anneal at 520 C for 7200 s only; got "
                f"{T:g} C for {hold:g} s. This is a limit of this instrument, not of the process")
    return steps


def _threshold_nm(context, sid):
    """coarse_number_fraction is defined only for a stated threshold (the agents' forecast contract requires it too)."""
    if context.get("coarse_threshold_nm") is None:
        raise ValueError(f"{sid}: coarse_number_fraction requires measurement_context.coarse_threshold_nm (a positive "
                         "threshold in nm); a coarse number fraction is defined only for a stated threshold")
    low, high = SUPPORT_ENVELOPE["coarse_threshold_nm"]
    return _number(context["coarse_threshold_nm"], f"{sid}: measurement_context.coarse_threshold_nm", low, high)


def validate_c7_request_controls(request, domain=None):
    """World-free validation of NEW requests to this family; raises ValueError. No world or package is read."""
    operation = request.get("operation", "manufacture")
    if operation not in {"manufacture", "anneal", "measure"}:
        raise ValueError("unsupported workflow operation")
    misplaced = sorted(set(request) & (CONTEXT_KEYS | {"measurement_context", "consumed_mass_g", "destructive",
                                                       "specimen_available"}))
    if misplaced:
        raise ValueError(f"{', '.join(misplaced)} belong on each specimen (material controls) or in "
                         "specimen.conditions.measurement_context (sampling), not on the request")
    process = {"anneal_steps", "temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa",
               "pressure_schedule", "pressure_full_at_c", "steps", "current_path", "feedstock_d50_um",
               "degassing_completeness"}
    if set(request) & process:
        raise ValueError("process controls belong in each specimen.conditions (or a preparation's conditions)")
    preparations = request.get("preparations", [])
    if operation != "manufacture" and preparations:
        raise ValueError("preparations are declared only with new manufacture")
    for preparation in preparations:
        _preparation(preparation.get("conditions"), str(preparation.get("preparation_id")))
    for specimen in request.get("specimens", []):
        sid = str(specimen.get("specimen_id"))
        condition = specimen.get("conditions")
        if not isinstance(condition, dict):
            raise ValueError(f"{sid}: each specimen requires a conditions object")
        if set(specimen) & CONTEXT_KEYS or set(condition) & (CONTEXT_KEYS - {"measurement_context"}):
            raise ValueError(f"{sid}: sampling choices belong in specimen.conditions.measurement_context")
        if set(condition) & {"consumed_mass_g", "destructive", "specimen_available"}:
            raise ValueError(f"{sid}: material controls belong directly on the specimen, not its conditions")
        if condition.get("operation", operation) != operation:
            raise ValueError(f"{sid}: specimen conditions.operation conflicts with the requested operation")
        selection = specimen.get("measurements", request.get("measurements"))
        if not isinstance(selection, list) or not selection:
            raise ValueError(f"{sid}: this laboratory requires an explicit nonempty measurements selection")
        refused = [r for r in (_refusal(q) for q in selection if q not in VOCABULARY) if r]
        if refused:
            raise ValueError(f"{sid}: " + "; ".join(refused))
        if len(selection) != len(set(selection)) or set(selection) - set(VOCABULARY):
            raise ValueError(f"{sid}: unsupported or duplicated measurements "
                             f"{sorted(set(selection) - set(VOCABULARY))}; vocabulary {sorted(VOCABULARY)}")
        if "sps_log" in selection and operation != "manufacture":
            raise ValueError(f"{sid}: an anneal or repeat measurement cannot produce a new SPS trace")
        context, _ = _context(condition, sid)
        if "coarse_threshold_nm" in context or "coarse_number_fraction" in selection:
            _threshold_nm(context, sid)
        if operation == "manufacture":
            _manufacture_controls(condition, sid)
        elif operation == "anneal":
            input_stage = request.get("input_stage", specimen.get("input_stage", "as_sintered"))
            _anneal_controls(condition, sid, input_stage, request.get("output_stage"))
        else:
            unknown = set(condition) - MEASURE_KEYS
            if unknown:
                raise ValueError(f"{sid}: a repeat measurement accepts only mass_g and measurement_context, "
                                 f"not {sorted(unknown)}")


# ------------------------------------------------------------------ history replay

def _history(history, sid):
    """Manufacture controls, executed preparation and executed anneal steps from bound released history."""
    if not isinstance(history, list) or not history:
        raise ValueError(f"{sid}: existing-specimen operation needs bound prior_processing_conditions")
    first = history[0]
    prep = first.get("executed_preparation")
    if first.get("operation") != "manufacture" or not isinstance(prep, dict) or "feedstock_d50_um" not in prep:
        raise ValueError(f"{sid}: processing history is not a manufacture record of this world's family; bodies made "
                         "in an earlier virtual world are kept as records and cannot be continued here")
    controls = {k: first.get(k) for k in ("temperature_c", "heating_rate_c_min", "hold_time_s", "pressure_mpa",
                                          "pressure_schedule", "pressure_full_at_c", "current_path")}
    executed = {k: prep.get(k) for k in ("method", "temperature_c", "hold_time_s", "feedstock_d50_um",
                                         "degassing_completeness")}
    steps, stage = [], first.get("output_stage", "as_sintered")
    for index, previous in enumerate(history[1:], start=1):
        if previous.get("operation") != "anneal" or not isinstance(previous.get("anneal_steps"), list):
            raise ValueError(f"{sid}: history entry {index} is not an executed anneal record")
        for step in previous["anneal_steps"]:
            steps.append((step["stage"], float(step["temperature_c"]), float(step["hold_time_s"])))
        stage = previous.get("output_stage", stage)
    return controls, executed, steps, stage


def _anneal_key(steps):
    """Anneal key of an executed step sequence (C7: the first step's hold may vary); any other sequence is outside
    this world."""
    seq = [(s, t, h) for s, t, h in steps]
    low, high = SUPPORT_ENVELOPE["first_anneal_hold_time_s"]
    if not seq:
        return "none"
    first_ok = seq[0][0] == "after_first_anneal" and seq[0][1] == 900.0 and low <= seq[0][2] <= high
    if len(seq) == 1 and first_ok:
        return "900"
    if len(seq) == 2 and first_ok and seq[1] == ("after_second_anneal", 520.0, SECOND_ANNEAL_HOLD_S):
        return "900+520"
    raise ValueError("not supported by this world: executed anneal history differs from the C7 anneal keys")


def _first_hold_h(steps):
    """Hold (h) of the executed 900 C step; None before any anneal."""
    return float(steps[0][2]) / 3600.0 if steps else None


def c7_route(controls, executed, anneal, hold_h=None):
    """Map companion controls and an executed preparation to the tool's route keys (README "Routes"); the same map
    as the C5 and C6 families, plus the executed 900 C hold (C7 anneal_hold_h). alignment is not a control here: every
    body is isotropic (the tool default 0.5)."""
    temperature = float(controls["temperature_c"])
    mode = SCHEDULES[controls["pressure_schedule"]]
    route = {"T_peak_C": temperature, "heating_rate_C_per_min": float(controls["heating_rate_c_min"]),
             "dwell_min": float(controls["hold_time_s"]) / 60.0, "P_MPa": float(controls["pressure_mpa"]),
             "pressure_mode": mode, "current_path": controls.get("current_path") or "compact", "anneal": anneal,
             "prep": "new", "feedstock_d50_um": float(executed["feedstock_d50_um"])}
    if mode == "cooling" and controls.get("pressure_full_at_c") is not None:
        route["load_ramp_min"] = (temperature - float(controls["pressure_full_at_c"])) / COOLING_RATE_C_MIN
    if executed["method"] == "none":
        route.update(degas=0.0, degas_completeness=1.0)
    else:
        route.update(degas=1.0, degas_completeness=executed["degassing_completeness"])
    if anneal != "none":
        route["anneal_hold_h"] = float(hold_h if hold_h is not None else 2.0)
    return route


# ------------------------------------------------------------------ package and world

_PACKAGES = {}


def load_package(package):
    """Load a named C6 package by directory; every file must match the world's sha256s. With a manifest the package's
    own load(manifest=...) check runs as well; without one (owner development snapshot) it loads unverified."""
    root = Path(package["dir"])
    want = package["sha256"]
    required = set(PACKAGE_FILES) - OPTIONAL_PACKAGE_FILES
    if not required <= set(want) or set(want) - set(PACKAGE_FILES):
        raise ValueError(f"world package must name the sha256 of {sorted(required)} (and optionally manifest)")
    for key in want:
        if _sha256_file(root / PACKAGE_FILES[key]) != want[key]:
            raise ValueError(f"package file {PACKAGE_FILES[key]} differs from the sha256 this world names")
    cache_key = tuple(sorted(want.items()))
    if cache_key not in _PACKAGES:
        name = "ndfeb_physics_tool_c7_" + want["tool"][:16]
        loader = SourceFileLoader(name, str(root / PACKAGE_FILES["tool"]))
        spec = importlib.util.spec_from_loader(name, loader)
        tool = importlib.util.module_from_spec(spec)
        loader.exec_module(tool)
        files = {k: root / PACKAGE_FILES[k] for k in ("physics", "forward", "posterior", "microstructure",
                                                     "coercivity_link", "anneal", "anneal_posterior")}
        if "manifest" in want:
            pm = tool.load(manifest=root / PACKAGE_FILES["manifest"], **files)
            if not pm.verified:
                raise ValueError("package manifest verification failed")
        else:
            pm = tool.load(verify=False, **files)
        if not pm._has_c6():
            raise ValueError("package carries no C6 microstructure layer (microstructure.py and the c6_* draws)")
        if pm.link is None:
            raise ValueError("package carries no coercivity link")
        if not pm._has_c7():
            raise ValueError("package carries no C7 anneal layer (anneal.py and anneal_posterior.npz)")
        _PACKAGES[cache_key] = (tool, pm)
    return _PACKAGES[cache_key]


def package_record(package_dir):
    """The package block of a world: directory, identity and the sha256 of every file. A package with a manifest
    (a built C6 package) is verified on every load; a development snapshot without one is recorded as unverified."""
    root = Path(package_dir).resolve()
    present = {k for k, n in PACKAGE_FILES.items() if (root / n).is_file()}
    missing = sorted(set(PACKAGE_FILES) - OPTIONAL_PACKAGE_FILES - present)
    if missing:
        raise ValueError(f"{root}: package lacks {[PACKAGE_FILES[k] for k in missing]}")
    record = {"dir": str(root), "sha256": {k: _sha256_file(root / PACKAGE_FILES[k]) for k in sorted(present)}}
    tool, pm = load_package(record)
    manifest = json.loads((root / PACKAGE_FILES["manifest"]).read_text()) if "manifest" in present else {}
    c6meta = pm.meta.get("c6", {})
    record.update(tool_version=manifest.get("tool_version", tool.TOOL_VERSION),
                  slice=manifest.get("slice", "C7"),
                  stage=manifest.get("stage", (c6meta.get("calibration") or {}).get("stage")),
                  data_cutoff_step=manifest.get("data_cutoff_step", c6meta.get("data_cutoff_step")),
                  manifest_verified=bool(pm.verified), link_version=pm.link.LINK_VERSION,
                  anneal_layer=pm.an_meta.get("version"))
    return record


def draw_world(package_dir, world_seed, world_id, measurement, drawn_at=None, lot_id=LOT_ID):
    """One hidden draw: mechanisms and stored-draw indices by the package's weights (its own _assign_bma and
    _assign_indicator, n = 1), from a generator seeded with the world seed, in the C5 family's order (so a C6 package
    that extends a C5 posterior gives the same C5 draw for the same seed); the C6 declared draws are those of the same
    coercivity index. J_GB is not drawn. Returns the owner world record."""
    package = package_record(package_dir)
    _, pm = load_package(package)
    rng = np.random.default_rng(int(world_seed))
    dlab, didx = pm._assign_bma("density", 1, rng)
    mlab, midx = pm._assign_bma("magnetic", 1, rng)
    lt = pm._assign_indicator("load_timing", mlab, midx, rng)
    dm, di, mm, mi = str(dlab[0]), int(didx[0]), str(mlab[0]), int(midx[0])
    amechs = [m for m in pm.an.ANNEAL_MECHS if m in pm.an_post]
    aw = np.array([pm.an_meta["mechanism_weights"][m] for m in amechs], float)
    am = amechs[int(rng.choice(len(amechs), p=aw / aw.sum()))]
    ai = int(rng.integers(0, len(next(iter(pm.an_post[am].values())))))
    declared = {"magnetic_index": {k: float(v[mi]) for k, v in sorted(pm.post[mm].items())
                                   if k.startswith(("c5_", "fd_"))},
                "density_index": {k: float(v[di]) for k, v in sorted(pm.post[dm].items()) if k.startswith("c5_")},
                "c6_index": {k: float(pm.c6post[mm][k][mi]) for k in pm.ms.C6_PARAMS},
                "anneal_index": {k: float(v[ai]) for k, v in sorted(pm.an_post[am].items())}}
    missing = [k for k in MEASUREMENT_KEYS if k not in measurement]
    if missing:
        raise ValueError(f"measurement block lacks {missing}")
    block = {k: float(measurement[k]) for k in MEASUREMENT_KEYS[:1]}
    low, high = LINK_DRAWS_RANGE
    n = measurement["link_model_draws"]
    if type(n) is not int or not low <= n <= high:
        raise ValueError(f"link_model_draws must be an integer in [{low}, {high}]")
    block["link_model_draws"] = n
    return {"family": FAMILY, "world_id": world_id, "package": package, "world_seed": int(world_seed),
            "draw": {"density": {"mechanism": dm, "index": di}, "magnetic": {"mechanism": mm, "index": mi},
                     "load_timing": str(lt[0]), "anneal": {"mechanism": am, "index": ai}},
            "declared_layer": declared, "lots": [str(lot_id)], "measurement": block,
            "provenance": {"drawn_at": drawn_at, "procedure": "numpy default_rng(world_seed); package "
                           "_assign_bma('density', 1), _assign_bma('magnetic', 1), _assign_indicator('load_timing') "
                           "in that order (as the C5 family); C6 declared draws c6_* at the coercivity index; then "
                           "the anneal mechanism by the anneal layer's weights and one of its stored draws; the "
                           "boundary-film magnetization is not drawn; world seed from secrets.randbits (owner "
                           "template, prepare_c7.py)"}}


class World:
    """One owner-held world: a fixed draw, its deterministic streams and the single-draw evaluation."""

    def __init__(self, world_config):
        if not isinstance(world_config, dict) or world_config.get("family") != FAMILY:
            raise ValueError("C7 campaign world requires an explicit ndfeb_c7_campaign_v1 world")
        unknown = set(world_config) - WORLD_KEYS
        if unknown:
            raise ValueError(f"unknown world keys {sorted(unknown)}")
        self.config = world_config
        self.tool, self.pm = load_package(world_config["package"])
        self.fw, self.ms, self.link = self.pm.fw, self.pm.ms, self.pm.link
        draw = world_config["draw"]
        self.dmech, self.didx = draw["density"]["mechanism"], int(draw["density"]["index"])
        self.mmech, self.midx = draw["magnetic"]["mechanism"], int(draw["magnetic"]["index"])
        self.lt = draw["load_timing"]
        if self.dmech not in self.pm.post or self.mmech not in self.pm.post or self.lt not in self.tool.LT_LEVELS:
            raise ValueError("world draw names a mechanism or explanation this package does not carry")
        for mech, idx in ((self.dmech, self.didx), (self.mmech, self.midx)):
            if not 0 <= idx < len(next(iter(self.pm.post[mech].values()))):
                raise ValueError("world draw index outside the package's stored draws")
        self.c6 = {k: np.array([float(self.pm.c6post[self.mmech][k][self.midx])]) for k in self.ms.C6_PARAMS}
        self.an = self.pm.an
        self.amech, self.aidx = draw["anneal"]["mechanism"], int(draw["anneal"]["index"])
        if self.amech not in self.pm.an_post or not 0 <= self.aidx < len(next(iter(self.pm.an_post[self.amech].values()))):
            raise ValueError("world draw names an anneal mechanism or index this package does not carry")
        self.an_p = {k: np.array([float(v[self.aidx])]) for k, v in self.pm.an_post[self.amech].items()}
        self.an_mech = np.array([self.amech])
        declared = world_config.get("declared_layer", {})
        expect = {"magnetic_index": {k: float(v[self.midx]) for k, v in self.pm.post[self.mmech].items()
                                     if k.startswith(("c5_", "fd_"))},
                  "density_index": {k: float(v[self.didx]) for k, v in self.pm.post[self.dmech].items()
                                    if k.startswith("c5_")},
                  "c6_index": {k: float(v[0]) for k, v in self.c6.items()},
                  "anneal_index": {k: float(v[0]) for k, v in sorted(self.an_p.items())}}
        if any(declared.get(k) != v for k, v in expect.items()):
            raise ValueError("world declared layer differs from the package's draws at the world's indices")
        self.seed = int(world_config["world_seed"])
        meas = world_config["measurement"]
        self.noise = {k: float(meas[k]) for k in MEASUREMENT_KEYS[:1]}
        self.link_n = int(meas["link_model_draws"])
        self.field_log_sd = float(self.fw.GRAIN_FIELD_LOG_SD)
        self.rho_s = float(self.tool.RHO_TD)
        self.z_hull = float(self.stream5("world", "z_hull").standard_normal())
        self._cache, self._links = {}, {}

    def _stream(self, family, identity):
        digest = hashlib.sha256(":".join((family, str(self.seed)) + tuple(map(str, identity))).encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def stream5(self, *identity):
        """C5-family streams (density, grain_size_nm, body, preparation, BSE class): the C5 family's own identities."""
        return self._stream(C5_STREAMS, identity)

    def stream(self, *identity):
        """Streams of the C6 quantities (the C6 family's namespace, so every C5 and C6 stream is that family's)."""
        return self._stream("ndfeb_c6_campaign_v1", identity)

    def stream7(self, *identity):
        """Streams of the C7 quantities (their own namespace)."""
        return self._stream(C7_STREAMS, identity)

    def piece_effect(self, specimen_id, piece_id):
        """A piece's density offset from its body (relative), drawn once per (world, specimen, piece): the same piece
        at two stages shares it."""
        return float(self.an_p["c7_sdpiece"][0] * self.stream7("piece", specimen_id, piece_id).standard_normal())

    def prep_z(self, lot_id, preparation_id):
        """Preparation effect z (x tau_prep of the density draw), drawn once per (world, lot, preparation)."""
        return float(self.stream5("prep", lot_id, preparation_id).standard_normal())

    def body(self, specimen_id):
        """Body effects, drawn once per (world, specimen): density deviation (declared sd, g/cm3) and the tool's
        specimen effect on ln Hci (sig_spec of the world's coercivity draw)."""
        r = self.stream5("body", specimen_id)
        rho_dev = float(r.normal(0.0, self.noise["density_body_sd_g_cm3"]))
        u = float(self.pm.post[self.mmech]["sig_spec"][self.midx] * r.standard_normal())
        return {"rho_dev_g_cm3": rho_dev, "u_spec": u}

    def evaluate(self, route, *, prep_z=0.0, rho_dev_g_cm3=0.0, u_spec=0.0):
        """Single-draw evaluation of the package at the world's indices: the tool's own route, density, degassing
        state and magnetic-latent functions with n = 1 (the same calls predict() makes for each of its draws). The C5
        block equals the C5 family's World.evaluate at the as-made density; the C6 additions are the as-sintered
        continuity W, the C5 oxide factor of a declared feedstock and the link's chemistry offset. C7: at an annealed
        route the body's stage density is its as-made density carried through the executed anneal by the world's
        anneal draw; Br is C5's times the body's gain over historical E's gain (900 C / 2 h) under that draw; the C5
        model grain follows the executed 900 C hold."""
        key = _canonical([route, prep_z, rho_dev_g_cm3, u_spec])
        if key in self._cache:
            return copy.deepcopy(self._cache[key])
        pm, fw, an = self.pm, self.fw, self.an
        r = pm._route(route)
        r.pop("_resolution", None)
        r["_degas_expected"] = pm._expected_state(r)
        A = pm._arrays(r)
        mu, _ = pm._density_mean(A, r, np.array([self.dmech]), np.array([self.didx]), np.array([[prep_z]]), "post")
        rho_route = float(mu[0])
        rho_body = float(np.clip(rho_route + rho_dev_g_cm3 / self.rho_s, 0.3, 1.02))
        state, eps = pm._degas_state(r, np.array([0.5]), self.stream5("world", "degas_extrapolation"))
        mlab, midx = np.array([self.mmech]), np.array([self.midx])
        lat = pm._magnetic_latent(A, r, mlab, midx, np.array([rho_body]), state, np.array([self.z_hull]),
                                  np.array([u_spec]), "post", eps=eps, indicators={"load_timing": np.array([self.lt])},
                                  sem=True)
        v = {k: float(np.asarray(x).reshape(-1)[0]) for k, x in lat.items()}
        hull = pm.hull(r, rho=rho_body)
        r.pop("_degas_expected", None)
        ox = pm._c6_oxide_factor(r, lat, mlab, midx, "post")
        W_as = v["W_bse"]
        W_ann = float(fw.bse_stage_continuity({"c5_rann": v["c5_rann"]}, W_as, "after_900_520"))
        bse = {"c5_Wstar": v["c5_Wstar"], "c5_tau": v["c5_tau"]}
        # C7 anneal layer at this route
        hold = r.get("anneal_hold_h")
        steps = [] if r["anneal"] == "none" else [(900.0, np.array([float(hold)]))] + (
            [(520.0, np.array([2.0]))] if r["anneal"] == "900+520" else [])
        rho_stage = float(an.anneal_density(self.an_p, self.an_mech, np.array([rho_body]), steps)[0]) if steps \
            else rho_body
        e = np.array([pm.fw.E_RHO])
        gain_E = float(an.anneal_density(self.an_p, self.an_mech, e, [(900.0, np.array([2.0]))])[0] / e[0])
        br = math.exp(v["lbr"]) * ((rho_stage / rho_body) / gain_E if steps else 1.0)
        grain = v["grain"]
        if steps:
            grain = float(pm._c7_grain_hold(lat, mlab, midx, "post", np.array([float(hold)]), r["T_peak_C"])[0])
        out = {"route": r, "rho_route_rel": rho_route, "rho_body_rel": rho_body,
               "rho_body_g_cm3": rho_body * self.rho_s, "rho_stage_rel": rho_stage,
               "rho_stage_g_cm3": rho_stage * self.rho_s, "anneal_gain_E": gain_E,
               "hci_kOe": math.exp(v["lh"]), "br_kGs": br,
               "squareness": 1.0 / (1.0 + math.exp(-v["sq"])), "grain_um": grain,
               "grain_stage_um": {"as_sintered": v["g_as_sintered"], "after_900_520": v["g_after_900_520"]},
               "W": {"as_sintered": W_as, "after_900_520": W_ann},
               "p_spread": {"as_sintered": float(fw.bse_p_spread(bse, np.array([W_as]))[0]),
                            "after_900_520": float(fw.bse_p_spread(bse, np.array([W_ann]))[0])},
               "c_m": v["c5_cm"], "lh": v["lh"],
               "out_of_hull": {m: bool(h["out_of_hull"]) for m, h in hull.items()},
               "oxide_factor": None if ox is None else float(np.asarray(ox).reshape(-1)[0]),
               "ln_chem_offset": v.get("ln_chem_offset", 0.0),
               "feedstock_declared": bool(pm._feedstock_declared(r))}
        self._cache[key] = out
        return copy.deepcopy(out)

    def body_state(self, lot_id, preparation_id, specimen_id, controls, executed, stage, hold_h=None):
        """Hidden state of one body at a stage: the same preparation and body effects at every stage (paired). The C6
        block is computed from the same body as made (as-sintered evaluation) and carried to the stage by the tool's
        stage maps; C7: the stage's own density and the executed 900 C hold (hold_h)."""
        b = self.body(specimen_id)
        z = self.prep_z(lot_id, preparation_id)
        state = self.evaluate(c7_route(controls, executed, STAGE_ANNEAL[stage], hold_h), prep_z=z, **b)
        made = self.evaluate(c7_route(controls, executed, "none"), prep_z=z, **b)
        state["tool_stage"] = TOOL_STAGE[stage]
        state["c6"] = self.c6_state(made, state)
        state["hold_h"] = hold_h
        return state

    def c6_state(self, made, state):
        """C6 latents of one body from its as-made C5 state: per tool stage the model grain D, the SEM-protocol median
        (before field sampling), the fracture intercept latent and the A3 latents; the section planarity and porosity
        latent; the link's inputs. C7: the body's current stage reads its own density (fracture feature factor,
        planarity, porosity, link Br) and its hold-scaled grain; the other stages are those of the as-made body."""
        ms, P = self.ms, self.c6
        cur = state["tool_stage"]
        rho_made = np.array([made["rho_body_rel"]])
        rho_cur = np.array([state["rho_stage_rel"]])
        W = np.array([made["W"]["as_sintered"]])
        ox = None if made["oxide_factor"] is None else np.array([made["oxide_factor"]])
        g = made["grain_stage_um"]
        D = {"as_sintered": g["as_sintered"], "after_900": g["after_900_520"], "after_900_520": g["after_900_520"]}
        if cur != "as_sintered":
            D[cur] = state["grain_um"]
        stages = {}
        for s in ms.C6_STAGES:
            Ds = np.array([D[s]])
            rs = rho_cur if s == cur else rho_made
            la = ms.a3_latent(P, W, s, ox)
            stages[s] = {"grain_model_um": float(D[s]), "grain_median_um": float(ms.grain_median(P, Ds)[0]),
                         "fracture_latent_um": float(ms.fracture_latent(P, Ds, rs)[0]),
                         "a3": {k: float(np.asarray(la[k]).reshape(-1)[0]) for k in ms.A3_KEYS}}
        return {"rho_rel": float(rho_cur[0]), "rho_made_rel": float(rho_made[0]), "W": float(W[0]),
                "planar": bool(ms.planar(rho_cur)[0]), "porosity_latent": float(ms.porosity_latent(rho_cur)[0]),
                "stages": stages, "ln_chem_offset": float(made["ln_chem_offset"]), "route": made["route"],
                "feedstock_declared": made["feedstock_declared"]}

    def link_flat_state(self, c6, tool_stage, density_rel=None):
        """The coercivity link's flat input for this body at one stage, n = link_model_draws copies of the body's
        hidden state: the same mapping as the tool's adapter applies per stage (model grain, sigma_ln, the stage's body
        density, A3 latents, as-sintered placement, no etched section, isotropic alignment, chemistry offset).
        density_rel: the density a magnetic reading is normalized by (default: the body's own stage density)."""
        n = self.link_n
        st, a3_as = c6["stages"][tool_stage], c6["stages"]["as_sintered"]["a3"]
        full = lambda x: np.full(n, float(x))
        return {"grain_median_um": full(st["grain_model_um"]), "grain_sigma_ln": full(self.c6["c6_sig"][0]),
                "density_rel": full(c6["rho_rel"] if density_rel is None else density_rel),
                "nd_rich_area_fraction": full(st["a3"]["nd_rich_area_fraction"]),
                "pocket_share": full(st["a3"]["pocket_share"]),
                "faint_line_coverage": full(st["a3"]["faint_line_coverage"]),
                "pocket_share_as_sintered": full(a3_as["pocket_share"]),
                "faint_line_coverage_as_sintered": full(a3_as["faint_line_coverage"]),
                "coverage_sem_lower": np.full(n, np.nan), "stage": np.full(n, tool_stage),
                "alignment_cos_theta": full(self.tool.ALIGN_DEFAULT), "ln_hci_chem_offset": full(c6["ln_chem_offset"])}

    def link_distribution(self, lot_id, preparation_id, specimen_id, c6, tool_stage, normalization_rel=None):
        """The link's model distribution for one body and stage: one call with its own deterministic generator per
        (world, body, stage), so repeated requests return the same distribution. J_GB is marginalized inside. C7: a
        stated normalization density replaces the body's own in Br and (BH)max (a VSM measures moment; volume
        quantities need a density), with the same generator, so only the normalization differs."""
        key = (lot_id, preparation_id, specimen_id, tool_stage, normalization_rel)
        if key not in self._links:
            out = self.link.link(self.link_flat_state(c6, tool_stage, normalization_rel),
                                 self.stream("link", lot_id, preparation_id, specimen_id, tool_stage))
            self._links[key] = _link_summary(out, self.link)
        return copy.deepcopy(self._links[key])


def _summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "q05": float(q[0]), "q25": float(q[1]),
            "median": float(q[2]), "q75": float(q[3]), "q95": float(q[4])}


def _link_summary(out, link):
    weights = {f"{J:g}": float(w) for J, w in link.JGB_POSTERIOR.items()}
    res = {}
    for k in LINK_OUTPUTS:
        s = _summary(out[k]["marginal"])
        s["J_GB_conditional_mean"] = {f"{J:g}": float(np.mean(a)) for J, a in sorted(out[k]["by_jgb"].items())}
        res[k] = s
    return {"outputs": res, "J_GB_posterior_weights": weights, "J_GB_units": "T",
            "flags_share_of_model_draws": {k: float(np.mean(v)) for k, v in out["flags"].items()},
            "n_model_draws": int(len(out["hci_kOe"]["marginal"])), "status": out["status"],
            "link_version": out.get("version", link.LINK_VERSION)}


# ------------------------------------------------------------------ instruments

def _record(value, unit, method, stage, **fields):
    return {"value": value, "unit": unit, "method": method, "status": "conditional_synthetic",
            "measurement_stage": stage, **fields}


def _unavailable(quantity, reason, stage, status="unavailable"):
    return {"value": None, "unit": UNITS.get(quantity, "mixed"), "status": status, "reason": reason,
            "measurement_stage": stage}


def _bse(world, state, stage, ident):
    """The C5 family's BSE class reading (spread or pooled), same stream and model; None after 900 C alone."""
    if stage == "after_first_anneal":
        return None
    p = state["p_spread"]["as_sintered" if stage == "as_sintered" else "after_900_520"]
    return "spread" if world.stream5("meas", "bse", *ident).random() < p else "pooled"


def _section_basis(world, rho):
    ms = world.ms
    if rho >= ms.RHO_PLANAR_OBSERVED:
        return "planar polished section (at or above 92 %TD, the least dense planar section on record)"
    return "planar polished section by declaration (80-92 %TD: no section on record)"


def _not_planar(quantity, world, stage, extra=""):
    reason = (f"not_planar: below {world.ms.RHO_PLANAR:.2f} relative density a polished section of this porous body "
              "is not planar (relief and particle edges, no matrix mode); no polished-section reading is returned"
              + extra)
    return _unavailable(quantity, reason, stage, status="not_planar")


def _a3_reading(world, c6, tool_stage, ident):
    """One polished-section reading of the five Nd-rich phase quantities (the tool's A3 measurement model)."""
    ms = world.ms
    latent = c6["stages"][tool_stage]["a3"]
    out = {}
    for k in ms.A3_KEYS:
        z = world.stream("meas", "a3", k, *ident).standard_normal()
        out[k] = float(ms.a3_measure(k, np.array([latent[k]]), np.array([z]))[0])
    return out


def _section(world, state, stage, ident, quantity):
    c6 = state["c6"]
    ms = world.ms
    label = _bse(world, state, stage, ident)
    reading = _a3_reading(world, c6, state["tool_stage"], ident)
    value = {"class": label, "description": BSE_TEXT.get(label),
             "nd_rich_phase": reading,
             "nd_rich_phase_units": {k: ("um" if k == "pocket_ecd_um" else "1") for k in ms.A3_KEYS},
             "nd_rich_phase_definitions": dict(world.pm._A3_BASIS),
             "section": _section_basis(world, c6["rho_rel"]), "detector": "backscattered electrons", "images": None,
             "status": "constructed_quantitative_reading_not_an_image",
             "observation_model": (
                 "class: the C5 declared BSE model (P(spread) = Phi((W - W*) / tau), W* and tau fixed per world; as "
                 "sintered and after 900 C/2 h + 520 C/2 h only). Nd-rich phase numbers: the C6 latents of the "
                 "world's draw at this body's continuity W and stage (the anneal is a declared paired transition of "
                 "the same body), read with the tool's declared body-and-field scatter; one reading per section"),
             "resolution_limit": ("SEM at 13-27 nm/px cannot see the 2-3 nm boundary film: the Nd-rich phase numbers "
                                  "bound the thick (resolvable) phase only"),
             "limits": [LIMIT, "Declared link functions conditioned on owner readings of RPI's images (tool README, "
                               "microstructure layer); not an RPI measurement", "No image is returned"]}
    if label is None:
        value["class_note"] = ("the C5 BSE class model covers as sintered and after 900 C/2 h + 520 C/2 h only; the "
                               "Nd-rich phase numbers after 900 C/2 h alone follow the declared share of the transition")
    if quantity == "boundary_chemistry":
        value["pocket_chemistry"] = {"value": None, "status": "not_modeled",
                                     "reason": "this world models no EDS chemistry of the pockets (oxide or metallic)"}
        value["film_magnetization"] = {"value": None, "status": "not_observable",
                                       "reason": FILM_STATEMENT + "; " + world.tool.REFUSALS["film_magnetization"]}
        value["statement"] = FILM_STATEMENT
    return value


def _sps_log(world, state, controls, ident_body):
    """The tool's descriptive SPS display trace (A4) for this body's schedule; its end state is the body's density."""
    ms = world.ms
    mass, diameter = controls["mass_g"], controls.get("diameter_mm", SPS_DIE_MM)
    if diameter != SPS_DIE_MM or not 0.2 <= mass <= 3.0:
        return None
    r = state["route"]
    rho = state["rho_body_rel"]
    prm = ms.sps_draw(1, world.stream("sps", *ident_body))
    tr = ms.sps_trace(r, np.array([rho]), prm, dt_s=SPS_DT_S, mass_g=mass, quantize=True)
    ev = {k: (float(np.asarray(v).reshape(-1)[0])) for k, v in tr["events"].items()}
    times = [float(t) for t in tr["t_s"]]
    temps = [round(float(t), 1) for t in tr["T_C"]]
    disp = [round(float(d), 2) for d in tr["displacement_mm"][0]]
    rel = [round(float(x), 5) for x in tr["rel_density"][0]]
    declared = r["pressure_mode"] != "cooling"
    status = ms.SPS_STATUS + ("; " + ms.SPS_STATUS_DECLARED_TIMING if declared else "")
    return {"time_s": times, "temperature_c": temps, "displacement_mm": disp, "relative_density": rel,
            "pressure_mpa": [None] * len(times),
            "channels": {"time_s": "s", "temperature_c": "deg C (sensor reading, 0.1 C display)",
                         "displacement_mm": "ram displacement, mm (increases with shrinkage; 0.01 mm display)",
                         "relative_density": "fraction of 7.62 g/cm3 (derived from the geometry)",
                         "pressure_mpa": "not recorded (RPI's display logs carry no pressure, force or current "
                                         "channel); null at every sample"},
            "events": {"t_heating_end_s": ev["t_heating_end_s"], "t_power_off_s": ev["t_power_off_s"],
                       "load_step_start_s": ev["step_start_s"], "load_step_duration_s": ev["step_duration_s"],
                       "load_step_displacement_mm": ev["step_dD_mm"], "load_step_share": ev["step_share"],
                       "pre_load_densification_points_TD": ev["pre_load_points_TD"],
                       "relative_density_at_450_c": ev["rho_450"], "shrinkage_onset_c": ev["onset_C"]},
            "end_relative_density": float(rho), "end_density_g_cm3": float(rho) * world.rho_s,
            "reduction": {"peak_temperature_c": max(temps), "final_displacement_mm": disp[-1],
                          "end_relative_density": float(rho), "sample_count": len(times)},
            "pressure_schedule": {v: k for k, v in SCHEDULES.items()}[r["pressure_mode"]],
            "geometry": {"die_diameter_mm": SPS_DIE_MM, "charge_mass_g": float(mass)},
            "status": status, "method": "descriptive_sps_display_trace",
            "limits": [LIMIT, "Descriptive trace: shape from declared ranges fitted to RPI's raw logs #1, #2, #4, #5; "
                              "the end state is this body's density; densification kinetics are not identified",
                       "No log above 800 C exists: the shape at higher peaks is an extrapolation",
                       "Sensor temperature need not be powder temperature; tooling expansion is part of the "
                       "displacement"]}


def _magnetic(quantity, dist, stage):
    """A magnetic key: the link's model distribution, value null (never a simulated measurement)."""
    if quantity == "demagnetization_curve":
        model = {"outputs": dist["outputs"],
                 "curve_form": ("intrinsic second quadrant J(H) = Br (1 - (|H| / Hci)^n), n = ln 0.1 / ln(Hk/Hci); "
                                "(BH)max from that curve, capped at Br^2 / 4 mu0"),
                 "units": {"hci_kOe": "kOe", "br_kGs": "kGs", "hk_over_hci": "1", "bhmax_MGOe": "MGOe"}}
        unit, method = "mixed", "coercivity_link_model_distribution_curve_parameters"
    else:
        model = dict(dist["outputs"][LINK_SOURCE[quantity]], unit=UNITS[quantity])
        unit, method = UNITS[quantity], "coercivity_link_model_distribution"
        if quantity == "hci_screen_kOe":
            model["note"] = "an open-circuit screen reads intrinsic coercivity: the link's Hci distribution"
    model.update({k: dist[k] for k in ("J_GB_posterior_weights", "J_GB_units", "flags_share_of_model_draws",
                                       "n_model_draws", "status", "link_version")})
    return {"value": None, "unit": unit, "method": method, "status": MAGNETIC_STATUS, "status_note": dist["status"],
            "measurement_stage": stage, "model_distribution": model, "reason": MAGNETIC_REASON, "uncertainty": None,
            "measurement_statistic": "model_distribution_not_a_measurement",
            "limits": [LIMIT, dist["status"], "Isotropic body (<cos theta> = 0.5) at room temperature",
                       "Material (demagnetization-corrected) quantities on the basis of RPI's historical DG11-3 "
                       "reduction (VSM, cube, N = 0.36 sum-one, H_int = H_app - N 4 pi M), which anchors the link; Hci "
                       "does not depend on the correction, while an uncorrected open-circuit reading would show lower "
                       "Br, Hk/Hci and (BH)max",
                       "A reading of Hci >= 20 kOe needs a reverse field well above 20 kOe after saturation (the "
                       "historical VSM ran to +/-90 kOe); no current-panel loop exists",
                       "J_GB_conditional_mean: the means at boundary-film magnetization 0, 0.6 and 1.2 T; the "
                       "distribution marginalizes them with J_GB_posterior_weights"]}


def simulate_c7_campaign(request, world_config, seed):
    """Native response entry point for manufacture, anneal and measure in a C7 campaign world."""
    if world_config.get("family") != FAMILY or request.get("origin") != "synthetic":
        raise ValueError("C7 campaign world requires an explicit synthetic family and request")
    world = World(world_config)
    validate_c7_request_controls(request)
    operation = request.get("operation", "manufacture")
    lot_id = request["lot_id"]
    if lot_id not in world_config["lots"]:
        raise ValueError(f"unknown powder lot {lot_id!r} in this world; this world's virtual lot is "
                         f"{world_config['lots'][0]!r}; bodies made in an earlier virtual world are kept as records "
                         "and cannot be continued here")
    request_hash = hashlib.sha256(_canonical(request)).hexdigest()
    executed_preparations, preparation_outcomes = {}, {}
    for preparation in request.get("preparations", []):
        pid = preparation["preparation_id"]
        executed = _preparation(preparation["conditions"], pid)
        executed_preparations[pid] = executed
        preparation_outcomes[pid] = {"executed": True, "conditions": executed,
                                     "requested_conditions": preparation["conditions"], "origin": "synthetic",
                                     "status": "conditional_execution_not_physical_preparation"}
    observations, delivered, outcomes = [], {}, {}
    for specimen in request["specimens"]:
        sid, pid, cycle = (specimen[k] for k in ("specimen_id", "preparation_id", "cycle_id"))
        condition = copy.deepcopy(specimen["conditions"])
        context, piece = _context(condition, sid)
        selection = specimen.get("measurements", request.get("measurements"))
        if operation == "manufacture":
            if "prior_processing_conditions" in specimen:
                raise ValueError(f"{sid}: manufacture cannot overwrite an existing specimen history")
            if pid not in executed_preparations:
                raise ValueError(f"{sid}: declare preparation {pid!r} with its conditions in request.preparations "
                                 "(identical to its first declaration)")
            executed = executed_preparations[pid]
            controls = _manufacture_controls(condition, sid)
            stage = "as_sintered"
            if request.get("output_stage", stage) != stage:
                raise ValueError(f"{sid}: manufacture produces as_sintered bodies")
            actual = {**condition, **controls, "executed_preparation": {"preparation_id": pid, **executed}}
            if "route" in condition:   # accepted for contract compatibility; this world reads the controls only
                actual["controls_without_effect"] = ["route"]
            steps = []
        else:
            controls, executed, steps, prior_stage = _history(specimen.get("prior_processing_conditions"), sid)
            if request.get("input_stage", prior_stage) != prior_stage:
                raise ValueError("request.input_stage conflicts with executed prior history")
            if specimen.get("input_stage", prior_stage) != prior_stage:
                raise ValueError("specimen.input_stage conflicts with executed prior history")
            actual = dict(condition)
            if operation == "anneal":
                new = _anneal_controls(condition, sid, prior_stage, request.get("output_stage"))
                steps = steps + [(s["stage"], float(s["temperature_c"]), float(s["hold_time_s"])) for s in new]
                stage = request["output_stage"]
                public = ("stage", "temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa",
                          "cooling_rate_c_min")
                actual["anneal_steps"] = [{k: s[k] for k in public} for s in new]
                actual["execution_control_status"] = "canonical_controls_from_executed_steps"
                actual["cooling_mode"] = condition.get("cooling_mode", "furnace_cooling")
                actual["controls_without_modeled_effect"] = [
                    "cooling_mode", "anneal_steps[].heating_rate_c_min", "anneal_steps[].cooling_rate_c_min",
                    "anneal_steps[].pressure_mpa"]
            else:
                stage = prior_stage
                if request.get("output_stage", stage) != stage:
                    raise ValueError("a measurement cannot change the processing stage")
                for k in ("input_stage", "process_stage"):   # C7: no stage key is accepted and ignored
                    if k in condition and condition[k] != stage:
                        raise ValueError(f"{sid}: conditions.{k} {condition[k]!r} conflicts with the body's executed "
                                         f"stage {stage!r}")
            if STAGE_ANNEAL[stage] != _anneal_key(steps):
                raise ValueError(f"{sid}: executed anneal history does not match stage {stage}")
        actual.update(operation=operation, output_stage=stage)
        hold_h = _first_hold_h(steps)
        state = world.body_state(lot_id, pid, sid, controls, executed, stage, hold_h)
        c6 = state["c6"]
        tool_stage = state["tool_stage"]
        identity = {"specimen_id": sid, "lot_id": lot_id, "preparation_id": pid, "cycle_id": cycle,
                    "origin": "synthetic", "process_stage": stage}
        ident = (sid, stage, request_hash[:16], piece)
        records = {}
        for quantity in selection:
            if quantity in REFUSED_IN_VOCABULARY:
                records[quantity] = _unavailable(quantity, "not supported by this world: "
                                                 + world.tool.REFUSALS[quantity], stage)
        if "density_g_cm3" in selection:   # C7: the piece's stage density read by the stated method
            an, ap = world.an, world.an_p
            method = context.get("density_method", "archimedes")
            if method not in an.DENSITY_METHODS:
                raise ValueError(f"{sid}: density_method {method!r} is not modelled by this world's package")
            eta = world.piece_effect(sid, piece)
            if hasattr(an, "piece_density"):  # C7.2; old packages retain their exact reading convention for replay
                rho_piece = an.piece_density(ap, world.an_mech, np.array([state["rho_body_rel"]]),
                                             np.array([eta]), [(t, h / 3600.0) for _, t, h in steps])
            else:
                rho_piece = np.array([state["rho_stage_rel"] + eta])
            mean = float(an.reading_mean(ap, rho_piece, method)[0])
            sd = float(an.reading_sd(ap, method)[0])
            value = (mean + sd * float(world.stream7("meas", "density", *ident).standard_normal())) * world.rho_s
            records["density_g_cm3"] = _record(
                value, "g/cm^3", f"conditional_{method}_density", stage, density_method=method, piece_id=piece,
                uncertainty={"reading_sd_g_cm3": sd * world.rho_s},
                uncertainty_basis=("the world's reading sd for this method (anneal layer); the body's deviation (sd "
                                   f"{world.noise['density_body_sd_g_cm3']:g} g/cm3) and the piece's offset from its "
                                   "body are part of the specimen, not of this reading"),
                method_model=("archimedes: the piece's density plus the share c7_kopen of its open porosity counted as "
                              "solid (unsealed immersion; 0 for a sealed or saturated-weight protocol, fixed per world)"
                              if method == "archimedes" else
                              "archimedes_saturated: dry, saturated and suspended weights; the bulk density of the "
                              "piece with the Archimedes reading sd, no open-porosity error"
                              if method == "archimedes_saturated" else
                              "geometric: mass over cylinder volume, relative bias c7_bgeo fixed per world (rough "
                              "faces, chipped edges and graphite foil overstate the volume)"),
                piece_note=("the same piece_id retains its initial local density offset; its local porosity evolves "
                            "through the full executed anneal history" if hasattr(an, "piece_density") else
                            "the same piece_id at two stages shares its offset from the body; different pieces do not"),
                measurement_statistic="single_piece_value", statistical_unit="specimen",
                measurement_population="whole_piece", limits=[LIMIT])
        if "grain_size_nm" in selection:   # the C5 family's model and stream
            e = float(world.stream5("meas", "grain", *ident).normal(0, world.field_log_sd))
            value = state["grain_um"] * state["c_m"] * math.exp(e) * 1000.0
            records["grain_size_nm"] = _record(
                value, "nm", "conditional_sem_mean_grain_size", stage,
                uncertainty={"field_sampling_log_sd": world.field_log_sd},
                uncertainty_basis="declared field sampling (log-sd) per reading; the method factor is fixed per "
                                  "world and not included",
                feature_definition="mean grain size as an SEM fracture-surface report gives it: C5 measurement "
                                   "model ln G = ln D + ln c_m + e (model grain D at this stage, method factor c_m "
                                   "fixed per world, field sampling e per reading)",
                measurement_statistic="reported_mean_grain_size", statistical_unit="specimen", surface="fracture",
                measurement_population="fracture_surface_fields", sampling_note="measurement_context sampling "
                "choices are recorded; the field-sampling scatter is the declared value whatever the field count",
                limits=[LIMIT, "Declared, uncalibrated measurement model (tool README, 'SEM and BSE outputs')"])
        if {"grain_d90_nm", "coarse_number_fraction"} & set(selection):
            ms, P = world.ms, world.c6
            # coherence (owner, Oct 18): the distribution is the one behind THIS request's grain_size_nm reading (the
            # C5 reported-mean basis, same stream and identity), with the C6 layer's sigma_ln; so d90 > mean always
            e5 = float(world.stream5("meas", "grain", *ident).normal(0, world.field_log_sd))
            mean_um = state["grain_um"] * state["c_m"] * math.exp(e5)
            sigma = float(P["c6_sig"][0])
            median = mean_um * math.exp(-sigma ** 2 / 2)
            common = dict(
                uncertainty=None,
                uncertainty_basis="the same field-sampled reading as grain_size_nm (declared field log-sd); the "
                                  "method factor and sigma_ln are fixed per world and not included",
                measurement_statistic="sem_grain_size_distribution_reading", statistical_unit="specimen",
                measurement_population="sem_fields", sampling_note="d90 and the coarse number fraction of one "
                "request come from the same field set; measurement_context sampling choices are recorded",
                limits=[LIMIT, "Log-normal distribution on the reported-mean basis of grain_size_nm (same reading), "
                               "with the width sigma_ln of the tool's microstructure layer (declared prior)",
                        "Fracture SE images of bodies below ~95 %TD show particles and ligaments, not grains"])
            if "grain_d90_nm" in selection:
                records["grain_d90_nm"] = _record(
                    float(ms.d90(np.array([median]), np.array([sigma]))[0]) * 1000.0, "nm",
                    "conditional_sem_grain_size_distribution_d90", stage,
                    feature_definition="d90 of the log-normal grain distribution whose mean is this request's "
                                       "grain_size_nm reading: median = mean x exp(-sigma^2/2), d90 = median x "
                                       "exp(1.2816 sigma); sigma fixed per world", **common)
            if "coarse_number_fraction" in selection:
                t_nm = _threshold_nm(context, sid)
                frac = float(ms.coarse_fraction(np.array([median]), np.array([sigma]), t_nm / 1000.0)[0])
                records["coarse_number_fraction"] = _record(
                    frac, "1", "conditional_sem_coarse_number_fraction", stage, coarse_threshold_nm=t_nm,
                    feature_definition=f"number fraction of grains larger than {t_nm:g} nm under the same "
                                       "log-normal distribution: Phi((ln median - ln threshold) / sigma)", **common)
        fracture = context.get("surface", "polished") == "fracture"
        for quantity in SECTION_KEYS:
            if quantity in selection and fracture:
                records[quantity] = _unavailable(
                    quantity, "not_quantitative_on_fracture: on a fracture surface topography, the fracture path and "
                    "detector geometry change BSE and SE brightness, so bright-area, pocket and porosity fractions are "
                    "not phase or pore measurements (an apparent change reversed under another contrast map in the P6 "
                    "review); request a planar polished section (surface 'polished')", stage,
                    status="not_quantitative_on_fracture")
        if "porosity_area_fraction" in selection and not fracture:
            if not c6["planar"]:
                records["porosity_area_fraction"] = _not_planar("porosity_area_fraction", world, stage)
            else:
                z = float(world.stream("meas", "porosity", *ident).standard_normal())
                value = min(c6["porosity_latent"] * math.exp(float(world.c6["c6_pbias"][0])
                                                             + world.ms.POROSITY_LOG_SD * z), 1.0)
                records["porosity_area_fraction"] = _record(
                    value, "1", "conditional_bse_dark_area_fraction_polished_section", stage, uncertainty=None,
                    uncertainty_basis=f"reading log-sd {world.ms.POROSITY_LOG_SD:g} (declared) plus a method bias "
                                      "fixed per world",
                    feature_definition="BSE dark (pore) area fraction of a planar polished section: (1 - rho) x "
                                       "exp(b + e), b the method bias fixed per world, e per reading; the dark fraction "
                                       "brackets rho-porosity only within a factor of 3-4",
                    section=_section_basis(world, c6["rho_rel"]), measurement_statistic="single_section_reading",
                    statistical_unit="specimen", limits=[LIMIT])
        for quantity in ("microstructure", "boundary_chemistry"):
            if quantity in selection and not fracture:
                if not c6["planar"]:
                    extra = ("; " + FILM_STATEMENT) if quantity == "boundary_chemistry" else ""
                    records[quantity] = _not_planar(quantity, world, stage, extra)
                else:
                    records[quantity] = _record(_section(world, state, stage, ident, quantity), "mixed",
                                                "conditional_bse_reading_polished_section", stage)
        if "sps_log" in selection:
            log = _sps_log(world, state, controls, (lot_id, pid, sid, request_hash[:16]))
            records["sps_log"] = (_record(log, "mixed", log["method"], stage) if log is not None else _unavailable(
                "sps_log", "not supported by this world: the descriptive SPS trace is defined for RPI's 10 mm die "
                "and charges of 0.2-3.0 g", stage))
        magnetic = [q for q in MAGNETIC if q in selection]
        if magnetic:
            norm = context.get("normalization_density_g_cm3")
            norm_rel = None if norm is None else float(norm) / world.rho_s
            dist = world.link_distribution(lot_id, pid, sid, c6, tool_stage, norm_rel)
            for quantity in magnetic:
                records[quantity] = _magnetic(quantity, dist, stage)
                records[quantity]["normalization"] = {
                    "density_g_cm3": c6["rho_rel"] * world.rho_s if norm is None else float(norm),
                    "basis": ("the body's own density at this stage (C7)" if norm is None else
                              "stated in measurement_context.normalization_density_g_cm3"),
                    "note": ("a magnetometer measures moment; Br and (BH)max per volume use a density. A density other "
                             "than the piece's own scales Br by their ratio and changes (BH)max; Hci does not depend on "
                             "it. RPI's historical DG11-3 reduction used 7.4 g/cm3 while its mass and dimensions give "
                             "6.60")}
        delivered[sid] = {**actual, **identity, "delivery_status": "declared_conditional_execution"}
        for quantity in selection:
            rec = records[quantity]
            observations.append({**rec, **identity, "observation_id": f"{request_hash[:16]}:{sid}:{quantity}",
                                 "quantity": quantity, "status": rec.get("status", "conditional_synthetic"),
                                 "uncertainty": rec.get("uncertainty"), "evidence_scope": EVIDENCE_SCOPE})
        consumed = _number(specimen.get("consumed_mass_g", 0), "consumed mass", 0)
        available_mass = condition.get("mass_g", specimen.get("remaining_mass_g"))
        if available_mass is not None:
            available_mass = _number(available_mass, "available mass", 0, 25)
            if consumed > available_mass:
                raise ValueError("declared consumption exceeds conditional specimen mass")
        remaining = None if available_mass is None else available_mass - consumed
        outcomes[sid] = {"executed": True, "stage": stage, "consumed_mass_g": consumed,
                         "destructive": specimen.get("destructive", False), "remaining_mass_g": remaining,
                         "specimen_available": specimen.get("specimen_available", remaining != 0),
                         "operation": operation, "preparation_execution_status": "conditional"}
    return {"delivery_id": "conditional-" + request_hash[:20], "origin": "synthetic", "complete": True,
            "observations": observations, "delivered_conditions": delivered,
            "specimen_outcomes": outcomes, "preparation_outcomes": preparation_outcomes,
            "manufactured_specimen_ids": list(delivered) if operation == "manufacture" else [],
            "simulation_accounting": {"operation": operation,
                                      "new_manufactured_count": len(delivered) if operation == "manufacture" else 0,
                                      "status": "conditional_not_physical_throughput"},
            "provenance": {"evidence_class": "conditional_synthetic", "model_version": FAMILY,
                           "physical_execution": False,
                           "request_world_seed_sha256": hashlib.sha256(_canonical(
                               {"request": request, "world": world_config, "seed": seed})).hexdigest()}}


# ------------------------------------------------------------------ agent-visible description

def public_capabilities(world_config):
    """Neutral, factual description of the laboratory. It names the package, never the draw or the seed."""
    package = world_config["package"]
    slice_name = package.get("slice") or "C6"
    supported = [q for q in VOCABULARY if q not in REFUSED_IN_VOCABULARY]
    return {
        "version": "c7-campaign-twin-v1", "selection": "simulation='magnetic'",
        "reference_id": f"campaign-world-{slice_name}", "world_family": FAMILY,
        "evidence_class": "conditional_computational", "physical_execution": False,
        "description": (
            f"Simulates a campaign world drawn from the owner-built physics tool {slice_name} (tool "
            f"{package['tool_version']}; data cut-off as stated in its README), which is Dynamical's model "
            "fitted to RPI's released records, with its microstructure layer and coercivity link. The world is one "
            "posterior draw of that tool: a density mechanism, a coercivity mechanism, a load-timing explanation, the "
            "microstructure-layer parameters, an anneal-densification mechanism and their values, fixed for this "
            "laboratory and hidden. Returns are conditional computation, not physical observations, and agreement with "
            "the tool is not independent validation of it. C7 (2026-10-23): SPS density updated on #7 and #8 as "
            "sintered; a load-free 900 C anneal can densify a body (calibrated on #8's one paired return, whose hold, "
            "method and piece pairing RPI has not stated); density readings name their method and piece."),
        "declared_and_uncalibrated": [
            "Feedstock particle size other than 4 um (feedstock_d50_um): grain, coercivity and density layers are "
            "priors that no record constrains (tool README, 'Feedstock particle size').",
            "SEM grain size (grain_size_nm): ln G = ln D + ln c_m + e, method factor c_m fixed per world, field "
            "sampling e (log-sd 0.15) per reading (tool README, 'SEM and BSE outputs').",
            "Grain-size distribution (grain_d90_nm, coarse_number_fraction): log-normal whose mean is the same "
            "reading as grain_size_nm (median = mean x exp(-sigma^2/2)); sigma_ln (prior 0.46-0.51) fixed per world "
            "(tool README, microstructure layer).",
            "BSE class reading: P(spread) = Phi((W - W*) / tau) from the world's continuity index W; W* and tau fixed "
            "per world; anchored only at owner readings of historical micrographs.",
            "Nd-rich phase numbers and their anneal change: declared link functions of the continuity W with "
            "declared reading scatter; the change after 900 C/2 h + 520 C/2 h is a declared paired transition of the "
            "same body; after 900 C/2 h alone the declared share of it (tool README, microstructure layer).",
            "sps_log: a descriptive trace; final density from the world's density; the densification kinetics are "
            "not identified; load timing other than on cooling is a declared placement of the load step.",
            "Degassing completeness other than the fitted states, complete degassing, an insulated current path and "
            "load from the start of the dwell are declared scenarios of the tool.",
            "Anneal densification (C7): four mechanisms (liquid-assisted for all porosity, closed porosity only, a "
            "bounded removable porosity, none) all fit #8's 7.07 -> 7.58 g/cm3 apart from the last; they disagree "
            "below about 90 %TD. The world holds one. Holds other than #8's unstated one, and E's 2 h magnetic anneal "
            "response at other holds, are declared.",
            "Density readings (C7): archimedes counts a fixed share of open porosity as solid (unsealed protocol "
            "unknown); geometric carries a fixed relative bias; each piece has its own offset from its body."],
        "limits": [
            "Magnetic outputs (hci_kOe, br_kGs, bhmax_MGOe, hci_screen_kOe, demagnetization_curve) are simulated and "
            "unscored in this window. Each returns the coercivity link's model distribution at the body's hidden "
            "microstructure (mean, sd, quantiles and the means conditional on a boundary-film magnetization of 0, 0.6 "
            "and 1.2 T) in model_distribution, with value null: no simulated measurement is returned.",
            "Film magnetization: not observable here. The 2-3 nm boundary film's magnetization is not observable by "
            "any measurement in this window, and no world holds a value of it.",
            "SEM at 13-27 nm/px cannot see the 2-3 nm boundary film: the Nd-rich phase numbers cover the thick "
            "(resolvable) phase only. No STEM, atom-probe or EDS chemistry is simulated.",
            "Polished-section readings (porosity_area_fraction, microstructure, boundary_chemistry) need a planar "
            "section: below 0.80 relative density they return status not_planar and no value; 0.80-0.92 is planar by "
            "declaration (no section on record).",
            "sps_log carries time, sensor temperature, ram displacement and derived relative density (no pressure "
            "channel) for RPI's 10 mm die and charges of 0.2-3.0 g.",
            "Outside the fitted records the world follows the tool's priors.",
            "Bodies made in an earlier virtual world remain readable records; they cannot be measured or annealed "
            "in this world.",
            "Every body is isotropic; magnetic distributions are material (demagnetization-corrected) quantities at "
            "room temperature, normalized by the body's own stage density unless measurement_context."
            "normalization_density_g_cm3 states another.",
            "Section readings (porosity_area_fraction, microstructure, boundary_chemistry) on surface 'fracture' return "
            "status not_quantitative_on_fracture; SEM grain readings are fracture-surface readings; planarity follows "
            "the body's density at its current stage."],
        "measurement_models": {
            "density_g_cm3": "the world's density at the route with the batch's preparation effect, plus a body "
                             f"deviation (sd {world_config['measurement']['density_body_sd_g_cm3']:g} g/cm3), carried "
                             "through any executed anneal, plus the piece's offset (measurement_context.piece_id), "
                             "read by measurement_context.density_method (archimedes, default, or geometric) with "
                             "that method's world-fixed bias and reading sd",
            "grain_size_nm": "C5 SEM model at the body's stage (as_sintered, after_first_anneal, after_second_anneal)",
            "grain_d90_nm, coarse_number_fraction": "the grain distribution behind the same reading as grain_size_nm at the body's "
                                                    "stage; coarse_number_fraction requires "
                                                    "measurement_context.coarse_threshold_nm (nm)",
            "porosity_area_fraction": "BSE dark area fraction of a planar polished section (1 - rho) x exp(b + e), "
                                      "b fixed per world, e per reading (log-sd 0.5)",
            "microstructure": "one planar-section BSE reading: 'Nd-rich phase spread along grain boundaries' or "
                              "'Nd-rich phase pooled at triple junctions' (as sintered or after the second anneal) "
                              "plus the Nd-rich area fraction, pocket size, pocket share, elongated share and "
                              "faint-line coverage",
            "boundary_chemistry": "the same section reading with 'film magnetization: not observable here'; no EDS "
                                  "pocket chemistry",
            "sps_log": "descriptive display trace for the requested schedule (manufacture only)",
            "hci_kOe, br_kGs, bhmax_MGOe, hci_screen_kOe, demagnetization_curve": "the coercivity link's model "
                                                                                  "distribution (value null)",
            "preparation_effect": "each new preparation_id draws its batch effect once in this world",
            "pairing": "measure and anneal on an existing body use that body's hidden state"},
        "supported_quantities": supported,
        "unsupported_in_this_world": {q: "not supported by this world: " + REFUSALS[q] for q in REFUSED_IN_VOCABULARY},
        "not_offered": {"film_magnetization": REFUSALS["film_magnetization"], "stem_eds": REFUSALS["stem_eds"]},
        "units": dict(UNITS),
        "controls": {
            "lot_id": world_config["lots"][0], "operations": ["manufacture", "measure", "anneal"],
            "manufacture": sorted(MANUFACTURE_KEYS),
            "required_manufacture": ["temperature_c", "heating_rate_c_min", "hold_time_s", "pressure_mpa",
                                     "pressure_schedule", "mass_g"],
            "pressure_schedules": {"on_heating": "load rises during heating and is held",
                                   "on_cooling": "load applied at the peak on cooling; pressure_full_at_c sets the "
                                                 "ramp at 100 C/min cooling (default ramp 0.3 min)",
                                   "on_dwell": "load rises from the start of the dwell (declared scenario)"},
            "current_path": list(CURRENT_PATHS),
            "preparation": ("Every manufacture request includes request.preparations for each preparation it uses, "
                            "including a reused one with identical conditions. conditions: method vacuum_degassing "
                            "(temperature_c, hold_time_s) or none; optional feedstock_d50_um (0.5-8.0 um, default "
                            "4.0) and degassing_completeness (number in [0, 1.5] or 'complete'; default 1 at 600 C for "
                            "600-900 s, required at other set-points)."),
            "anneal": ("Existing body; canonical anneal_steps: after_first_anneal 900 C for 1800-14400 s (7200 s is "
                       "the planned program; #8's hold was not stated), after_second_anneal 520 C for 7200 s; "
                       "cooling_mode furnace_cooling or argon_quench (recorded, no modeled effect)"),
            "measurement": sorted(MEASURE_KEYS), "measurement_context": sorted(CONTEXT_KEYS),
            "measurement_context_c7": {"density_method": list(DENSITY_METHODS), "surface": list(SURFACES),
                                       "normalization_density_g_cm3": "number in [4, 8] g/cm3; default the body's own "
                                                                      "stage density",
                                       "piece_id": "names the piece; the same piece at two stages is paired"},
            "support_envelope": {k: (list(v) if isinstance(v, tuple) else v) for k, v in SUPPORT_ENVELOPE.items()}},
        "resources": "No specimen or request ceiling is configured for this virtual laboratory."}


def preview_task():
    return {"anneal_model_domain": PREVIEW_DOMAIN, "measurement_capabilities": list(VOCABULARY)}
