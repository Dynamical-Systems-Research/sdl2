"""Versioned conditional NdFeB process models; never a calibrated RPI twin.

Source reconstruction, cross-route transfer hypotheses and virtual observations
are distinct. The kinetic closures are inspectable assumptions. No coefficient
below is inferred from RPI measurements. See the science source-to-model record.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

MODEL_VERSION = "conditional-rpi-workflow-v1"
R = 8.31446261815324
REFERENCE_K = 1023.15
_QUAD_X, _QUAD_W = np.polynomial.legendre.leggauss(48)
HYPOTHESES = ("mobile_boundary", "contact_limited", "boundary_blocked")
RPI_INFORMED_ANNEAL_DOMAIN = {
    "after_first_anneal_max_temperature_c": 900.,
    "after_second_anneal_max_temperature_c": 600.,
    "max_step_hold_time_s": 7200.,
    "status": "declared_conditional_support_envelope_not_a_material_growth_threshold",
}


def thermal_dose(
    temperature_c: float,
    heating_rate_c_min: float,
    hold_time_s: float,
    activation_energy_kj_mol: float,
    *,
    offset_c: float = 0,
    start_c: float = 25,
    cooling_rate_c_min: float | None = None,
) -> float:
    """Equivalent seconds at 750 C for a linear ramp, dwell, optional cooling.

The offset is applied to the complete material temperature history. Zero-duration
ramp at start_c is valid. A missing cooling rate omits cooling, not a claim of
instantaneous physical cooling. This is not an energy or electricity estimate.
    """
    values = (temperature_c, heating_rate_c_min, hold_time_s,
              activation_energy_kj_mol, offset_c, start_c)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Thermal history must be finite")
    if temperature_c < start_c or heating_rate_c_min <= 0 or hold_time_s < 0:
        raise ValueError("Require temperature >= start, positive ramp, nonnegative dwell")
    if activation_energy_kj_mol < 0 or min(start_c, temperature_c) + offset_c <= -273.15:
        raise ValueError("Require nonnegative activation energy and positive Kelvin temperature")
    if cooling_rate_c_min is not None and (
        not math.isfinite(cooling_rate_c_min) or cooling_rate_c_min <= 0
    ):
        raise ValueError("Cooling rate must be finite and positive")
    q_over_r = activation_energy_kj_mol * 1000 / R
    span = temperature_c - start_c
    temperatures_k = start_c + span * (_QUAD_X + 1) / 2 + offset_c + 273.15
    average_factor = float(np.dot(_QUAD_W, np.exp(
        -q_over_r * (1 / temperatures_k - 1 / REFERENCE_K)
    ))) / 2
    ramp_seconds = span / heating_rate_c_min * 60
    cooling_seconds = 0 if cooling_rate_c_min is None else span / cooling_rate_c_min * 60
    hold_factor = math.exp(-q_over_r * (1 / (temperature_c + offset_c + 273.15)
                                       - 1 / REFERENCE_K))
    return (ramp_seconds + cooling_seconds) * average_factor + hold_time_s * hold_factor


def _source_rows() -> list[dict[str, Any]]:
    with (Path(__file__).parent / "data" / "density.csv").open() as handle:
        return list(csv.DictReader(handle))


def _fit_density(rows: list[dict[str, Any]], contact: bool, initial_relative: float) -> dict:
    observed = np.array([float(r["density_g_cm3"]) for r in rows])
    rates = np.array([float(r["heating_rate_c_min"]) for r in rows])
    rho0 = initial_relative * 7.57

    def predict(x: np.ndarray) -> np.ndarray:
        rho_inf, log_k, q = x[:3]
        gamma = x[3] if contact else 0.0
        dose = np.array([thermal_dose(float(r["temperature_c"]),
                                     float(r["heating_rate_c_min"]),
                                     float(r["hold_time_s"]), q) for r in rows])
        dose *= (rates / 100) ** gamma
        return rho_inf - (rho_inf - rho0) * np.exp(-math.exp(log_k) * dose)

    lows = [7.54, -15, 40] + ([-2] if contact else [])
    highs = [7.65, 2, 500] + ([4] if contact else [])
    starts = [[7.58, -3, q] + ([1.0] if contact else []) for q in (100, 240, 400)]
    fits = [least_squares(lambda x: predict(x) - observed, x, bounds=(lows, highs),
                          max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            for x in starts]
    fit = min(fits, key=lambda f: float(np.sum(f.fun ** 2)))
    estimated = predict(fit.x)
    residuals = estimated - observed
    return {
        "source_id": "ikram_2019", "stage": "initial_sps", "n_rows": len(rows),
        "n_unique_histories": len({r["history_id"] for r in rows}),
        "n_fitted_parameters": len(fit.x), "initial_relative_density_assumed": initial_relative,
        "rho_infinity_g_cm3": float(fit.x[0]), "rate_per_s_at_750": math.exp(fit.x[1]),
        "activation_energy_kj_mol": float(fit.x[2]),
        "ramp_contact_exponent": float(fit.x[3]) if contact else 0.0,
        "rmse_g_cm3": float(np.sqrt(np.mean(residuals ** 2))),
        "max_abs_residual_g_cm3": float(np.max(np.abs(residuals))),
        "residuals": [{"row_id": r["row_id"], "observed_g_cm3": float(y),
                       "predicted_g_cm3": float(p), "residual_g_cm3": float(e)}
                      for r, y, p, e in zip(rows, observed, estimated, residuals)],
        "fit_status": "conditional_source_reconstruction_not_mechanism_identification",
        "success": bool(fit.success),
        "bound_active": [name for name, x, lo, hi in zip(
            ["rho_infinity", "log_rate", "activation_energy", "ramp_contact_exponent"],
            fit.x, lows, highs) if min(abs(x - lo), abs(x - hi)) < 1e-4],
    }


@lru_cache(maxsize=1)
def fit_source_models() -> dict:
    """Execute source fits and direct falsification checks; no physical validation.

The table's rounding and unspecified manufacture n preclude likelihood-based
confidence claims. Initial packing profiles expose a nuisance not fitted by data.
    """
    rows = _source_rows()
    pre = [r for r in rows if r["source_id"] == "ikram_2019" and r["post_treatment"] == "none"]
    pairs = []
    for r in pre:
        post = next(p for p in rows if p["source_id"] == "ikram_2019"
                    and p["history_id"] == r["history_id"] and p["post_treatment"] != "none")
        pairs.append({"history_id": r["history_id"],
                      "temperature_c": float(r["temperature_c"]),
                      "delta_density_g_cm3": float(post["density_g_cm3"]) - float(r["density_g_cm3"])})
    thermal = _fit_density(pre, False, .65)
    contact = _fit_density(pre, True, .65)
    grain_initial = 240.0
    growth_ratio = (750**2 - grain_initial**2) / (330**2 - grain_initial**2)

    def ratio_residual(q: float) -> float:
        return thermal_dose(850, 100, 60, q) / thermal_dose(750, 100, 60, q) - growth_ratio

    grain_q = brentq(ratio_residual, 1, 800)
    grain_k = (330**2 - grain_initial**2) / thermal_dose(750, 100, 60, grain_q)
    return {
        "model_version": MODEL_VERSION,
        "density_thermal_only": thermal,
        "density_ramp_contact": contact,
        "initial_packing_profiles": [
            _fit_density(pre, contact_model, initial)
            for contact_model in (False, True) for initial in (.55, .75)
        ],
        "paired_anneal_changes": pairs,
        "grain_conditional_fit": {
            "source_id": "ikram_2019", "n_anchor_means": 2,
            "source_locator": "journal p1200; initial powder p1205 summary",
            "grain_initial_nm_assumed_source_summary": grain_initial,
            "growth_exponent_assumed": 2.0, "activation_energy_kj_mol": grain_q,
            "growth_nm2_per_s_at_750": grain_k,
            "fitted_mean_grain_nm": [330.0, 750.0], "temperatures_c": [750.0, 850.0],
            "recipe_assumed": {"heating_rate_c_min": 100, "hold_time_s": 60},
            "status": "two_point_exact_reconstruction_conditional_on_initial_size_exponent_and_recipe",
            "prediction_sd": None,
            "source_conflicts": [
                "p1205 conclusion alternatively states 400+/-250 nm and a 750-2600 nm bimodal population",
                "Initial-powder summary 240+/-100 nm differs from p1199 fracture-surface 240-420 nm range",
                "Methods say 50 C/min whereas Table2/figure discussion use 100 C/min for preferred history",
            ],
        },
        "direct_model_failure": {
            "contrast": "700 C, 60 s, 50 MPa: ramp 50 minus ramp 100 C/min",
            "observed_delta_density_g_cm3": 7.36 - 7.49,
            "thermal_only_predicted_sign": "nonnegative_for_positive_rate",
            "conclusion": "thermal-dose-only closure cannot reproduce this negative contrast; unknown repeatability prevents statistical rejection",
        },
        "source_separation": ["Mishra and Keszler constrain plausibility separately; no pooled fit",
                              "Wu only constrains population/stage semantics, no kinetic fit",
                              "Ikram anneal 750 C/15 min is not RPI 900 C/2 h + 520 C/2 h"],
    }


def _parameters(hypothesis: str) -> dict:
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis {hypothesis}; choose {HYPOTHESES}")
    fit = fit_source_models()
    density = fit["density_ramp_contact" if hypothesis == "contact_limited" else "density_thermal_only"]
    grain = fit["grain_conditional_fit"]
    return {
        "density_activation_energy_kj_mol": density["activation_energy_kj_mol"],
        "density_rate_per_s_at_750": density["rate_per_s_at_750"],
        "ramp_contact_exponent": density["ramp_contact_exponent"],
        "density_transfer_multiplier": .01 if hypothesis == "boundary_blocked" else 1.0,
        "limiting_relative_density": .965 if hypothesis == "boundary_blocked" else .995,
        "initial_relative_density": .65,
        "solid_density_g_cm3": 7.58,
        "pressure_exponent": 1.0,
        "temperature_offset_c": 0.0,
        "grain_activation_energy_kj_mol": grain["activation_energy_kj_mol"],
        "grain_growth_nm2_per_s_at_750": grain["growth_nm2_per_s_at_750"],
        "grain_transfer_multiplier": 1.0,
        "initial_fine_grain_nm": 240.0,
        "initial_coarse_grain_nm": 1000.0,
        "coarse_region_area_fraction": .15,
        "coarse_growth_multiplier": 2.0,
        "grain_limit_nm": None,
        "anneal_density_multiplier": .01,
        "anneal_grain_multiplier": 1.0,
        "preparation_grain_multiplier": 1.0,
        "furnace_cooling_rate_c_min": 5.0,
        "argon_quench_rate_c_min": 200.0,
        "anneal_heating_rate_c_min": 10.0,
    }


def parameter_basis() -> dict:
    """Explicit sensitivity ranges; neither priors nor experimentally fitted CIs."""
    return {
        "density_transfer_multiplier": {"range": [.001, 1.0], "basis": "cross-route mobility stress range; unmeasured Ames lot"},
        "temperature_offset_c": {"range": [-25., 25.], "basis": "conditional indicated-to-material offset; no RPI calibration claim"},
        "solid_density_g_cm3": {"range": [7.50, 7.65], "basis": "brackets source denominators 7.55/7.57/7.63; unknown target chemistry"},
        "initial_relative_density": {"range": [.55, .75], "basis": "packing sensitivity, not a measured preparation result"},
        "pressure_exponent": {"range": [0., 2.], "basis": "unidentified pressure sensitivity; sources do not isolate pressure"},
        "initial_fine_grain_nm": {"range": [50., 400.], "basis": "conditional fine starting population; Wu and Ikram histories differ"},
        "initial_coarse_grain_nm": {"range": [500., 5000.], "basis": "heterogeneous/residual population scenario; not an Ames measurement"},
        "coarse_region_area_fraction": {"range": [.05, .40], "basis": "sampling/population stress range, not transferred Wu area fractions"},
        "grain_transfer_multiplier": {"range": [.05, 2.], "basis": "unknown target grain mobility relative to conditional Ikram fit"},
        "anneal_density_multiplier": {"range": [0., .05], "basis": "pressureless closure remains unidentified by paired density changes"},
        "anneal_grain_multiplier": {"range": [.01, 1.], "basis": "distinct SPS/furnace mechanisms; Ikram only observes mild growth at 750 C/15 min"},
        "grain_limit_nm": {"range": [800., 2000.], "basis": "optional pinning ceiling alternative; null means unpinned; not a fitted Zener law"},
        "furnace_cooling_rate_c_min": {"range": [2., 10.], "basis": "conditional cooling profile; accepted plan unresolved"},
        "argon_quench_rate_c_min": {"range": [50., 300.], "basis": "conditional cooling profile; not actual heat-transfer prediction"},
    }


def _resolved_anneal_steps(condition: dict, parameters: dict, prior_stage: str, stage: str,
                          cooling_rate: float) -> list[dict]:
    """Resolve an explicit, pressureless two-stage schedule without ignored controls.

Each custom step must name its stage, temperature, dwell seconds and heating
rate. Top-level numeric controls, if present, describe the FIRST executed step
and must agree. Defaults are only the accepted 900/520 C, 7200 s per-step cycle.
Legacy aliases are never accepted here, including aliases equal to the default.
    """
    if "steps" in condition:
        raise ValueError("Unsupported anneal control 'steps'; submit canonical anneal_steps with "
                         "stage, temperature_c, hold_time_s and heating_rate_c_min for each step")
    metadata = {"operation", "input_stage", "output_stage", "process_stage", "cooling_mode",
                "mass_g", "remaining_mass_g", "route", "lot_id", "origin", "preparation_id",
                "cycle_id", "specimen_id", "delivery_status", "treatment_cycle_id",
                "post_treatment", "requested_post_treatment_label", "measurement_context",
                "geometry_status", "mass_status", "diameter_mm", "evidence_origin",
                "evidence_status", "execution_control_status", "execution_deviation",
                "executed_as_requested", "treatment_cycle_id_semantics", "history_basis"}
    controls = {"anneal_steps", "temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa"}
    unknown = set(condition) - metadata - controls
    if unknown:
        raise ValueError(f"Unsupported anneal controls {sorted(unknown)}; express process controls "
                         "inside canonical anneal_steps, not unrecognized aliases")
    if prior_stage not in ("as_sintered", "after_first_anneal"):
        raise ValueError("Anneal input stage must be as_sintered or after_first_anneal")
    if condition.get("input_stage", prior_stage) != prior_stage:
        raise ValueError("Anneal input_stage conflicts with executed prior history")
    if condition.get("process_stage", stage) not in (prior_stage, stage):
        raise ValueError("Anneal process_stage conflicts with input/output stages")
    expected_stages = ([] if prior_stage == "after_first_anneal" else ["after_first_anneal"])
    if stage == "after_second_anneal":
        expected_stages.append("after_second_anneal")
    defaults = [{"stage": name, "temperature_c": 900. if name == "after_first_anneal" else 520.,
                 "hold_time_s": 7200., "heating_rate_c_min": parameters["anneal_heating_rate_c_min"]}
                for name in expected_stages]
    custom = "anneal_steps" in condition
    steps = condition["anneal_steps"] if custom else defaults
    if not isinstance(steps, list) or len(steps) != len(expected_stages) or not steps:
        raise ValueError(f"anneal_steps must contain exactly the ordered stages {expected_stages}")
    required = {"stage", "temperature_c", "hold_time_s", "heating_rate_c_min"}
    allowed = required | {"pressure_mpa", "cooling_rate_c_min"}
    resolved = []
    for index, (step, expected_stage) in enumerate(zip(steps, expected_stages, strict=True)):
        if not isinstance(step, dict) or set(step) - allowed or required - set(step):
            raise ValueError("Each anneal_steps item requires stage, temperature_c, hold_time_s, "
                             "heating_rate_c_min; only pressure_mpa=0 and cooling_rate_c_min are optional")
        if step["stage"] != expected_stage:
            raise ValueError(f"anneal_steps stage/order conflict at index {index}; expected {expected_stage}")
        actual = dict(step)
        actual.setdefault("pressure_mpa", 0.)
        actual.setdefault("cooling_rate_c_min", cooling_rate)
        for key in allowed - {"stage"}:
            value = actual[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"anneal_steps {key} must be a finite number")
        if actual["temperature_c"] < 25 or actual["hold_time_s"] <= 0:
            raise ValueError("anneal_steps requires temperature_c >= 25 and positive hold_time_s")
        if actual["heating_rate_c_min"] <= 0 or actual["cooling_rate_c_min"] <= 0:
            raise ValueError("anneal_steps heating and cooling rates must be positive")
        if actual["pressure_mpa"] != 0:
            raise ValueError("The anneal model is pressureless; anneal_steps pressure_mpa must be 0")
        resolved.append(actual)
    for key in controls - {"anneal_steps"}:
        if key in condition:
            value = condition[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value != resolved[0][key]:
                if not custom:
                    raise ValueError(f"Nondefault anneal {key} requires canonical anneal_steps; "
                                     "top-level controls cannot override the accepted schedule")
                raise ValueError(f"Top-level anneal {key} conflicts with the first canonical anneal_steps item")
    if not custom and "post_treatment" in condition:
        labels = {"anneal_" + "_".join(
            f"{step['temperature_c']:g}c_{step['hold_time_s']/divisor:g}{unit}"
            for step in resolved) for divisor, unit in ((1, "s"), (60, "min"), (3600, "h"))}
        if condition["post_treatment"] not in labels:
            raise ValueError("An ambiguous post_treatment label cannot specify an anneal; "
                             "submit canonical anneal_steps (labels alone cannot override controls)")
    return resolved


def validate_anneal_condition(condition: dict, *, input_stage: str, output_stage: str) -> None:
    """Validate public anneal controls without fitting or evaluating a model.

Defaults name the accepted schedule and 10 C/min heating. Owner-specific rates
may still be used when no top-level rate is submitted; any requested alternative
must be explicit in anneal_steps. This validates controls, not history authority.
    """
    if output_stage not in ("after_first_anneal", "after_second_anneal"):
        raise ValueError("Anneal stage must be after_first_anneal or after_second_anneal")
    if input_stage == output_stage:
        raise ValueError("Requested anneal stage is already complete")
    if condition.get("cooling_mode", "furnace_cooling") not in ("furnace_cooling", "argon_quench"):
        raise ValueError("Select furnace_cooling or argon_quench as an explicit branch")
    _resolved_anneal_steps(condition, {"anneal_heating_rate_c_min": 10.},
                          input_stage, output_stage, 5.)


def validate_rpi_informed_anneal_domain(
    condition: dict, *, input_stage: str, output_stage: str,
) -> None:
    """World-free public support check for the new RPI-informed furnace branch.

    The envelope limits a conditional approximation, not actual process safety or
    a universal growth threshold. Legacy extrapolation calls do not use it.
    """
    validate_anneal_condition(condition, input_stage=input_stage, output_stage=output_stage)
    steps = _resolved_anneal_steps(condition, {"anneal_heating_rate_c_min": 10.},
                                  input_stage, output_stage, 5.)
    for step in steps:
        maximum = RPI_INFORMED_ANNEAL_DOMAIN[f"{step['stage']}_max_temperature_c"]
        if step["temperature_c"] > maximum or step["hold_time_s"] > 7200.:
            raise ValueError(
                f"RPI-informed anneal response is unsupported for {step['stage']} above "
                f"{maximum:g} C or dwell above 7200 s; no bulk/boundary response is claimed"
            )


def conditional_forecast(
    condition: dict[str, Any], hypothesis: str = "mobile_boundary",
    parameters: dict[str, Any] | None = None, prior_state: dict[str, Any] | None = None,
    *, anneal_response: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Advance one specimen's conditional state through SPS or selected anneals.

Condition keys follow the existing campaign: temperature_c, pressure_mpa,
heating_rate_c_min, hold_time_s. operation=anneal requires prior_state. The
after_second_anneal stage executes the accepted two anneals unless the prior is
already after_first_anneal. Optional anneal_steps allows explicit counterfactuals.
    """
    if condition.get("evidence_origin") == "physical":
        raise ValueError("A conditional forecast may not be labeled physical evidence")
    if prior_state is not None and prior_state.get("model_version") != MODEL_VERSION:
        raise ValueError("Prior state model version mismatch")
    previous_response = prior_state.get("anneal_response") if prior_state else None
    if prior_state is not None and previous_response is None and anneal_response is not None:
        raise ValueError("An anneal response must be selected before specimen manufacture")
    if previous_response is not None:
        if anneal_response is not None and anneal_response != previous_response:
            raise ValueError("An existing specimen cannot change its frozen anneal response")
        anneal_response = previous_response
    if anneal_response is not None:
        anneal_response = dict(anneal_response)
        allowed = {"mode", "bulk_response", "boundary_morphology"}
        if set(anneal_response) - allowed:
            raise ValueError("Unknown anneal_response fields")
        if anneal_response.get("mode") not in {"sps_growth_extrapolation", "boundary_phase_dominant"}:
            raise ValueError("Select an explicit supported anneal response mode")
        if anneal_response.get("bulk_response") not in {"retained", "mobility_sensitivity"}:
            raise ValueError("Declare retained or mobility_sensitivity bulk response")
        if anneal_response.get("boundary_morphology") not in {"unresolved", "contrast_redistribution"}:
            raise ValueError("Declare unresolved or contrast_redistribution boundary morphology")
        if (anneal_response["mode"] == "sps_growth_extrapolation"
            and anneal_response["bulk_response"] != "mobility_sensitivity"):
            raise ValueError("SPS extrapolation requires explicit mobility_sensitivity semantics")
    p = _parameters(hypothesis)
    if prior_state:
        p.update(prior_state["parameters"])
        hypothesis = prior_state["hypothesis"]
    if anneal_response and anneal_response["bulk_response"] == "retained":
        # A zero-change limiting hypothesis, not a fitted physical bound. Neither
        # Lian's expectation nor the paired 750 C HDDR results identify 900 C kinetics.
        p.update(anneal_density_multiplier=0., anneal_grain_multiplier=0.)
    if parameters:
        unknown = set(parameters) - set(p)
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}")
        p.update(parameters)
    if (anneal_response and anneal_response["bulk_response"] == "retained"
        and (p["anneal_density_multiplier"] != 0 or p["anneal_grain_multiplier"] != 0)):
        raise ValueError("Retained bulk response requires zero anneal density and grain mobility")
    if (anneal_response and anneal_response["mode"] == "boundary_phase_dominant"
        and anneal_response["bulk_response"] == "mobility_sensitivity" and not previous_response
        and not {"anneal_density_multiplier", "anneal_grain_multiplier"} <= set(parameters or {})):
        raise ValueError("Boundary-dominant sensitivity requires explicit uncalibrated anneal mobilities")
    for name, value in p.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Nonfinite parameter {name}")
    if not 0 < p["initial_relative_density"] < p["limiting_relative_density"] <= 1:
        raise ValueError("Initial density must be below limiting relative density <= 1")
    if not 0 <= p["coarse_region_area_fraction"] <= 1:
        raise ValueError("Area fraction outside [0, 1]")
    positive = ("solid_density_g_cm3", "density_rate_per_s_at_750", "initial_fine_grain_nm",
                "initial_coarse_grain_nm", "anneal_heating_rate_c_min",
                "furnace_cooling_rate_c_min", "argon_quench_rate_c_min")
    if any(p[k] <= 0 for k in positive):
        raise ValueError("Densities, grain sizes, rates must be positive")
    nonnegative = ("density_activation_energy_kj_mol", "density_transfer_multiplier",
                   "grain_activation_energy_kj_mol", "grain_growth_nm2_per_s_at_750",
                   "grain_transfer_multiplier", "coarse_growth_multiplier", "anneal_density_multiplier",
                   "anneal_grain_multiplier", "preparation_grain_multiplier")
    if any(p[k] < 0 for k in nonnegative) or (p["grain_limit_nm"] is not None and p["grain_limit_nm"] <= 0):
        raise ValueError("Mobility parameters must be nonnegative and pinning limit positive")
    operation = condition.get("operation", "manufacture")
    stage = condition.get("output_stage", condition.get("process_stage", "as_sintered"))
    stage = "as_sintered" if stage == "initial_sps" else stage
    if operation not in ("manufacture", "anneal", "measure"):
        raise ValueError("Operation must be manufacture, anneal or measure")
    if operation in ("anneal", "measure") and prior_state is None:
        raise ValueError("Existing-specimen operation requires prior_state")
    if operation == "manufacture" and prior_state is not None:
        raise ValueError("Manufacture cannot overwrite an existing specimen history")
    rho_s = float(p["solid_density_g_cm3"])
    rho = float(prior_state["density_g_cm3"]) if prior_state else rho_s * p["initial_relative_density"]
    fine = float(prior_state["fine_region_mean_grain_size_nm"]) if prior_state else p["initial_fine_grain_nm"]
    coarse = float(prior_state["coarse_region_mean_grain_size_nm"]) if prior_state else p["initial_coarse_grain_nm"]
    history = list(prior_state["history"]) if prior_state else []
    steps = []
    cooling_mode = condition.get("cooling_mode", "furnace_cooling")
    if cooling_mode not in ("furnace_cooling", "argon_quench"):
        raise ValueError("Select furnace_cooling or argon_quench as an explicit branch")
    cool_rate = p["furnace_cooling_rate_c_min"] if cooling_mode == "furnace_cooling" else p["argon_quench_rate_c_min"]
    if operation == "manufacture":
        # Preparation shares genealogy across temperature groups; density closure
        # is not active without compaction. Only conditional grain exposure enters.
        if condition.get("include_preparation", True):
            steps.append({"stage": "prepared", "temperature_c": 600., "heating_rate_c_min": 5.,
                          "hold_time_s": 900., "pressure_mpa": 0., "cooling_rate_c_min": p["furnace_cooling_rate_c_min"]})
        steps.append({"stage": "as_sintered", "temperature_c": float(condition["temperature_c"]),
                      "heating_rate_c_min": float(condition.get("heating_rate_c_min", 10)),
                      "hold_time_s": float(condition.get("hold_time_s", 300)),
                      "pressure_mpa": float(condition.get("pressure_mpa", 60)),
                      "cooling_rate_c_min": condition.get("sps_cooling_rate_c_min", cool_rate)})
    elif operation == "anneal":
        if stage not in ("after_first_anneal", "after_second_anneal"):
            raise ValueError("Anneal stage must be after_first_anneal or after_second_anneal")
        if prior_state["process_stage"] == "after_second_anneal":
            raise ValueError("Accepted anneal sequence is already complete")
        if prior_state["process_stage"] == "after_first_anneal" and stage == "after_first_anneal":
            raise ValueError("First anneal is already complete")
        if anneal_response and anneal_response["mode"] == "boundary_phase_dominant":
            validate_rpi_informed_anneal_domain(
                condition, input_stage=prior_state["process_stage"], output_stage=stage)
        steps = _resolved_anneal_steps(condition, p, prior_state["process_stage"], stage, cool_rate)
    else:
        stage = prior_state["process_stage"]
    totals = {"density_equivalent_s_at_750": 0., "grain_equivalent_s_at_750": 0.}
    for step in steps:
        if step["pressure_mpa"] < 0:
            raise ValueError("Pressure must be nonnegative")
        args = {"temperature_c": step["temperature_c"],
                "heating_rate_c_min": step["heating_rate_c_min"], "hold_time_s": step["hold_time_s"],
                "offset_c": p["temperature_offset_c"], "cooling_rate_c_min": step["cooling_rate_c_min"]}
        jd = thermal_dose(**args, activation_energy_kj_mol=p["density_activation_energy_kj_mol"])
        jg = thermal_dose(**args, activation_energy_kj_mol=p["grain_activation_energy_kj_mol"])
        totals["density_equivalent_s_at_750"] += jd
        totals["grain_equivalent_s_at_750"] += jg
        if step["stage"] == "prepared":
            density_multiplier = 0.
            grain_multiplier = p["preparation_grain_multiplier"]
        elif step["stage"] == "as_sintered":
            density_multiplier = ((step["pressure_mpa"] / 50) ** p["pressure_exponent"]
                                  * (step["heating_rate_c_min"] / 100) ** p["ramp_contact_exponent"])
            grain_multiplier = 1.
        else:
            density_multiplier = p["anneal_density_multiplier"]
            grain_multiplier = p["anneal_grain_multiplier"]
        ceiling = max(rho, rho_s * p["limiting_relative_density"])
        rho = ceiling - (ceiling - rho) * math.exp(
            -p["density_rate_per_s_at_750"] * p["density_transfer_multiplier"] * density_multiplier * jd)
        growth = p["grain_growth_nm2_per_s_at_750"] * p["grain_transfer_multiplier"] * grain_multiplier * jg
        fine_new = math.sqrt(fine**2 + growth)
        coarse_new = math.sqrt(coarse**2 + p["coarse_growth_multiplier"] * growth)
        if p["grain_limit_nm"] is not None:
            # A declared pinning alternative cannot shrink a previously larger grain.
            fine_new = min(fine_new, max(fine, p["grain_limit_nm"]))
            coarse_new = min(coarse_new, max(coarse, p["grain_limit_nm"] * 2))
        fine, coarse = fine_new, coarse_new
        history.append({**step, "temperature_offset_c": p["temperature_offset_c"],
                        "density_equivalent_s_at_750": jd, "grain_equivalent_s_at_750": jg,
                        "density_g_cm3": rho, "fine_region_mean_grain_size_nm": fine,
                        "coarse_region_mean_grain_size_nm": coarse})
    porosity = max(0., 1 - rho / rho_s)
    latent = {"schema_version": 1, "model_version": MODEL_VERSION, "hypothesis": hypothesis,
              "parameters": p, "process_stage": stage, "density_g_cm3": rho,
              "porosity_fraction": porosity, "fine_region_mean_grain_size_nm": fine,
              "coarse_region_mean_grain_size_nm": coarse,
              "coarse_region_area_fraction": p["coarse_region_area_fraction"], "history": history}
    if anneal_response is not None:
        latent["anneal_response"] = anneal_response
        anneals = [h for h in history if h["stage"] == "after_first_anneal"]
        first = anneals[0] if anneals else None
        # Categorical fixtures are specified only for the accepted first anneal
        # and the published paired protocol. No interpolated boundary kinetics.
        specified = first is not None and (
            (first["temperature_c"], first["hold_time_s"]) in {(900., 7200.), (750., 900.)}
        )
        latent["boundary_morphology_state"] = (
            "as_sintered_reference" if first is None else
            "boundary_contrast_pattern_changed" if specified and anneal_response["boundary_morphology"]
            == "contrast_redistribution" else "unresolved"
        )
    result = {
        "model_version": MODEL_VERSION, "evidence_status": "conditional_computation",
        "hypothesis": hypothesis, "process_stage": stage, "density_g_cm3": rho,
        "relative_density_pct": 100 * rho / rho_s, "porosity_fraction": porosity,
        "fine_region_mean_grain_size_nm": fine, "coarse_region_mean_grain_size_nm": coarse,
        "coarse_region_area_fraction": p["coarse_region_area_fraction"],
        "grain_size_nm": fine, "grain_measurement_population": "fine_region",
        "thermal_dose": totals, "latent_state": latent, "chemistry": None, "magnetics": None,
        "assumptions": [
            "Ames powder identity and preparation are not equated with MQU-F or recycled HDDR",
            "Kinetics transferred conditionally from Ikram; actual RPI calibration absent",
            "Named grain populations are conditional descriptors; no micrograph was measured",
            "Coarse area fraction and grain transfer/pinning are explicit assumptions",
            "Thermal dose is not energy use, throughput, cost or magnetic performance",
            "Anneal closure extrapolates beyond 750 C/15 min source treatment",
            "Cooling rates and material-temperature offset are assumptions until actual logs",
            "Each anneal is conditionally cooled to 25 C before the next heating; interstage path is unconfirmed",
            "Ikram grain body/conclusion and recipe descriptions conflict; kinetics are assumption-dominated",
            "Three shared preparation groups confound preparation and accepted temperature",
        ],
        "uncertainty": {"status": "scenario_sensitivity_not_probability_or_confidence_interval",
                        "source_fit_residual_is_measurement_sd": False,
                        "physical_repeatability_sd": None, "parameter_basis": parameter_basis()},
    }
    if anneal_response is not None:
        result["anneal_response"] = dict(anneal_response)
        result["boundary_phase_response"] = {
            "morphology_state": latent["boundary_morphology_state"],
            "phase_fraction": None, "boundary_thickness_nm": None,
            "phase_identity": None, "coercivity": None,
            "status": "conditional_qualitative_scenario_not_calibrated_boundary_kinetics",
            "required_evidence": "paired registered SEM/BSE fields with sampling and contrast metadata; "
                "EDS or higher resolution phase analysis with instrument limits for chemical assignment; "
                "separate magnetic measurements for functional response",
        }
        result["assumptions"].append(
            "Explicit furnace response separates the bulk growth hypothesis from boundary morphology; "
            "retention is a limiting case, sensitivity mobilities are uncalibrated, and qualitative "
            "contrast cannot establish boundary chemistry or magnetic improvement"
        )
    return result


def forecast_panel(parameters: dict | None = None, include_anneals: bool = True) -> dict:
    rows = []
    for hypothesis in HYPOTHESES:
        for temperature in (750, 800, 850):
            condition = {"temperature_c": temperature, "pressure_mpa": 60,
                         "heating_rate_c_min": 10, "hold_time_s": 300}
            pre = conditional_forecast(condition, hypothesis, parameters)
            rows.append({"temperature_c": temperature, "stage": "as_sintered", **pre})
            if include_anneals:
                for cooling in ("furnace_cooling", "argon_quench"):
                    post = conditional_forecast(
                        {"operation": "anneal", "output_stage": "after_second_anneal", "cooling_mode": cooling},
                        hypothesis, prior_state=pre["latent_state"])
                    rows.append({"temperature_c": temperature, "stage": "after_second_anneal",
                                 "cooling_mode": cooling, **post})
    return {"model_version": MODEL_VERSION, "replicates_per_temperature": 3,
            "shared_preparation_groups": 3, "independent_temperature_preparation_cells": 3,
            "replicate_noise_not_fitted": True, "forecasts": rows}


def discriminant_report() -> dict:
    """Executed contrasts linked to measurements, not an imposed decision rule."""
    contrast = []
    for hypothesis in HYPOTHESES:
        vals = {}
        for label, temp, ramp, dwell in (("reference", 800, 10, 300), ("lower_temperature", 750, 10, 300),
                                        ("faster_ramp", 800, 100, 300), ("shorter_dwell", 800, 10, 60)):
            f = conditional_forecast({"temperature_c": temp, "heating_rate_c_min": ramp,
                                      "hold_time_s": dwell, "pressure_mpa": 60}, hypothesis)
            vals[label] = {k: f[k] for k in ("density_g_cm3", "fine_region_mean_grain_size_nm", "porosity_fraction")}
        contrast.append({"hypothesis": hypothesis, "predictions": vals,
                         "faster_ramp_minus_reference_density_g_cm3": vals["faster_ramp"]["density_g_cm3"] - vals["reference"]["density_g_cm3"],
                         "faster_ramp_minus_reference_grain_nm": vals["faster_ramp"]["fine_region_mean_grain_size_nm"] - vals["reference"]["fine_region_mean_grain_size_nm"]})
    return {"counterfactuals": contrast,
            "measurement_requirements": [
                "Dry/immersed masses and wetting/temperature procedure separate porosity from Archimedes bias",
                "Actual SPS temperature/pressure/time and a tooling correction constrain delivered-dose interpretation",
                "Registered fine/coarse fields and section locations compare grain populations across stages",
                "Same specimen before/after anneals tests history retention; later magnetic measurement tests function",
                "Matched preparation split between follow-up process settings separates preparation from temperature",
            ],
            "not_identified": ["magnetic performance", "specific boundary chemistry", "energy/cost optimum",
                               "causal temperature effect from three confounded preparation groups"]}
