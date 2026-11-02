"""Owner-side conditional worlds. Never call these outputs physical measurements.

Each world declares response curves and noise assumptions. These are stress-test
worlds, not a fit to undisclosed RPI specimens or calibrated instrument noise.
No filesystem or delivery-release mutation occurs in this module.
"""

import hashlib
import json
import math

import numpy as np

from .forecast import finite
from .microstructure import simulate_microstructure


def synthetic_worlds() -> dict:
    """Explicitly constructed alternatives; no default winning intervention.

    Returned values are fresh copies. Selection of a world belongs to the owner,
    and its config should not be sent to the investigator as acquired evidence.
    """
    common = {
        "evidence_class": "conditional_synthetic",
        "route": "mqu_f_sps",
        "assumption_rationale": "Public SPS density plateau/temperature tradeoff motivates "
        "constructed alternative worlds; not a physical parameter fit",
        "temperature_range_c": [550.0, 800.0],
        "reference_pressure_mpa": 50.0,
        "requested_control_bounds": {
            "pressure_mpa": [50.0, 50.0],
            "hold_time_s": [30.0, 30.0],
            "heating_rate_c_min": [100.0, 100.0],
            "mass_g": [15.0, 15.0],
            "diameter_mm": [20.0, 20.0],
        },
        "reference_hold_time_s": 30.0,
        "reference_heating_rate_c_min": 100.0,
        "reference_density_g_cm3": 7.63,
        "density_measurement_sd_g_cm3": 0.01,
        "density_preparation_sd_g_cm3": 0.025,
        "density_cycle_sd_g_cm3": 0.015,
        "temperature_readback_sd_c": 2.0,
        "temperature_bias_c": 0.0,
        "pressure_readback_sd_mpa": 0.5,
        "grain_measurement_sd_nm": 8.0,
        "grain_preparation_sd_nm": 10.0,
        "grain_cycle_sd_nm": 5.0,
        "grain_boundaries_resolved": True,
        "initial_temperature_c": 25.0,
        "cooling_rate_c_min": 100.0,
        "tool_expansion_mm_per_c": 0.0003,
        "green_relative_density": 0.55,
        "density_pressure_sensitivity_g_cm3_per_mpa": 0.0,
        "density_hold_sensitivity_g_cm3_per_s": 0.0,
        "density_ramp_sensitivity_g_cm3_per_c_min": 0.0,
        "noise_status": "explicit_design_assumptions_not_estimated_from_sources",
    }
    curves = {
        "density_tradeoff": {
            "density_curve": [[550, 6.35], [600, 6.95], [650, 7.34], [700, 7.49], [800, 7.51]],
            "grain_curve": [[550, 55], [600, 65], [650, 90], [700, 140], [800, 320]],
        },
        "density_plateau": {
            "density_curve": [[550, 7.42], [600, 7.46], [650, 7.49], [700, 7.50], [800, 7.50]],
            "grain_curve": [[550, 65], [600, 75], [650, 100], [700, 155], [800, 340]],
        },
        "no_microstructure_benefit": {
            "density_curve": [[550, 6.30], [600, 6.80], [650, 7.23], [700, 7.48], [800, 7.49]],
            "grain_curve": [[550, 130], [600, 130], [650, 130], [700, 130], [800, 130]],
        },
    }
    return {name: {**common, **specific, "world_id": name} for name, specific in curves.items()}


def microstructure_stress_world() -> dict:
    """Explicitly constructed adverse-tail alternative, not an RPI clone.

    Literature motivates measuring heterogeneity, not these numerical parameters.
    All original worlds remain unchanged. The new operation is opt-in.
    """
    world = synthetic_worlds()["density_plateau"]
    world.update(world_id="distribution_tradeoff_v1", die_material="graphite",
                 temperature_readback_sd_c=0.0, pressure_readback_sd_mpa=0.0)
    # This version replaces the legacy scalar grain law with feature sampling.
    # Do not carry unused scalar grain noise parameters as if they were active.
    for key in ("grain_curve", "grain_preparation_sd_nm", "grain_cycle_sd_nm"):
        world.pop(key)
    world["grain_measurement_sd_nm"] = 0.0  # Not used as uncertainty in extended observations.
    world["microstructure"] = {
        "assumption_rationale": "Constructed counterexample: a smaller mean can hide a coarse "
        "tail and increased section porosity. No fit to RPI or paper feature data.",
        "field_count": 4, "features_per_field": 50, "feature_cv": .12,
        "porosity_field_concentration": 500.0,
        "coarse_threshold_nm": 250.0,
        "fine_size_curve_nm": [[550, 15], [600, 20], [650, 70], [700, 110], [800, 190]],
        "coarse_size_curve_nm": [[550, 400], [600, 350], [650, 160], [700, 170], [800, 350]],
        "coarse_component_weight_curve": [[550, .18], [600, .12], [650, .04], [700, .02], [800, .1]],
        "porosity_area_fraction_curve": [[550, .04], [600, .025], [650, .016], [700, .015], [800, .015]],
    }
    return world


def _curve(config, key, temperature):
    pairs = config.get(key)
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError(f"{key} requires at least two explicit temperature/value pairs")
    pairs = sorted((finite(x, key), finite(y, key)) for x, y in pairs)
    if len({x for x, _ in pairs}) != len(pairs) or any(y <= 0 for _, y in pairs):
        raise ValueError(f"Invalid {key} points")
    if not pairs[0][0] <= temperature <= pairs[-1][0]:
        raise ValueError(f"Delivered temperature outside declared {key}; no extrapolation")
    return float(np.interp(temperature, *zip(*pairs, strict=True)))


def _sd(config, key):
    value = finite(config.get(key, 0.0), key)
    if value < 0:
        raise ValueError(f"{key} cannot be negative")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rng(seed, identity):
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _sps_log(condition, config, density):
    """Simple declared ramp/hold/cool readback model, not a thermal FE solution."""
    initial = finite(config.get("initial_temperature_c", 25.0), "initial_temperature_c")
    peak = condition["temperature_c"]
    ramp = condition["heating_rate_c_min"]
    dwell = condition["hold_time_s"]
    cooling = finite(config.get("cooling_rate_c_min", 100.0), "cooling_rate_c_min")
    if ramp <= 0 or cooling <= 0 or peak <= initial:
        raise ValueError("Positive ramps and peak above initial temperature are required")
    rise = (peak - initial) * 60.0 / ramp
    cool = (peak - initial) * 60.0 / cooling
    times = np.unique(
        np.r_[
            np.linspace(0, rise, 9),
            np.linspace(rise, rise + dwell, 5),
            np.linspace(rise + dwell, rise + dwell + cool, 9),
        ]
    )
    temperatures = np.interp(
        times, [0, rise, rise + dwell, rise + dwell + cool], [initial, peak, peak, initial]
    )
    mass, diameter = condition.get("mass_g"), condition.get("diameter_mm")
    displacement = [None] * len(times)
    if mass is not None and diameter is not None:
        mass, diameter = finite(mass, "mass_g"), finite(diameter, "diameter_mm")
        if mass <= 0 or diameter <= 0:
            raise ValueError("Mass and diameter must be positive")
        reference = finite(config["reference_density_g_cm3"], "reference_density_g_cm3")
        green = finite(config.get("green_relative_density", 0.55), "green_relative_density")
        if not 0 < green <= 1:
            raise ValueError("green_relative_density must be in (0,1]")
        area = math.pi * (diameter / 2.0) ** 2
        h0 = mass / (reference * green) * 1000.0 / area
        hf = mass / density * 1000.0 / area
        # Measured displacement includes irreversible shrinkage and reversible tooling expansion.
        progress = np.minimum(times / max(rise + dwell, 1e-9), 1.0)
        expansion = finite(config.get("tool_expansion_mm_per_c", 0.0), "tool_expansion")
        displacement = ((h0 - hf) * progress - expansion * (temperatures - initial)).tolist()
    return {
        "time_s": times.tolist(),
        "temperature_c": temperatures.tolist(),
        "pressure_mpa": [condition["pressure_mpa"]] * len(times),
        "displacement_mm": displacement,
        "status": "synthetic_readback_not_actual_execution",
        "model": "declared_linear_ramp_hold_cool_plus_tool_expansion",
        "temperature_meaning": "indicated_temperature; actual internal field unmodeled",
    }


def simulate_batch(request: dict, world_config: dict, seed: int) -> dict:
    """Return an un-released delivery matching the operator store contract.

    Measurement selection: each specimen can provide measurements=["density_g_cm3",
    "grain_size_nm","sps_log"]. Without it, density+SPS log only. Grain requires
    explicit selection and a world declaration that boundaries are resolved.
    """
    if request.get("origin") != "synthetic":
        raise ValueError("Simulator accepts only explicitly synthetic requests")
    if world_config.get("evidence_class") != "conditional_synthetic":
        raise ValueError("world_config must declare conditional_synthetic evidence")
    if not world_config.get("assumption_rationale"):
        raise ValueError("world_config must state its conditional assumptions")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lot = request.get("lot_id")
    specimens = request.get("specimens")
    if not isinstance(lot, str) or not lot or not isinstance(specimens, list) or not specimens:
        raise ValueError("Synthetic request requires a lot and nonempty specimens")
    content_hash = hashlib.sha256(
        _canonical({"request": request, "world": world_config, "seed": seed})
    ).hexdigest()
    observations, delivered, cycle_recipes = [], {}, {}
    for specimen in specimens:
        sid = specimen.get("specimen_id")
        prep, cycle = specimen.get("preparation_id"), specimen.get("cycle_id")
        if not all(isinstance(s, str) and s for s in (sid, prep, cycle)) or sid in delivered:
            raise ValueError("Distinct specimen IDs with preparation and cycle IDs required")
        condition = dict(specimen.get("conditions", {}))
        if condition.get("route") != world_config.get("route"):
            raise ValueError("World and requested powder routes differ")
        if condition.get("post_treatment") != "none":
            raise ValueError("Synthetic worlds currently cover no post-treatment only")
        if condition.get("process_stage", "initial_sps") != "initial_sps":
            raise ValueError("Synthetic worlds cover initial SPS only")
        if condition.get("die_material") is not None and (
            world_config.get("die_material") is None
            or condition["die_material"] != world_config["die_material"]
        ):
            raise ValueError("Synthetic world has no support for requested die_material")
        for key, expected in (
            ("measurement_population", "selected_sections" if "microstructure" in world_config else "synthetic_descriptor"),
            ("measurement_statistic", "number_mean" if "microstructure" in world_config else "constructed_scalar_descriptor"),
        ):
            if key in condition and condition[key] != expected:
                raise ValueError(f"Synthetic world does not represent requested {key}")
        condition["process_stage"] = "initial_sps"
        if condition.get("lot_id", lot) != lot:
            raise ValueError("Condition lot differs from requested lot")
        condition.update(lot_id=lot, preparation_id=prep, cycle_id=cycle)
        for key in ("temperature_c", "pressure_mpa", "hold_time_s", "heating_rate_c_min"):
            condition[key] = finite(condition.get(key), key)
            if condition[key] <= 0:
                raise ValueError(f"{key} must be positive")
        controls = world_config.get("requested_control_bounds")
        if not isinstance(controls, dict):
            raise ValueError("World needs explicit requested_control_bounds")  # noqa: TRY004
        for key in ("pressure_mpa", "hold_time_s", "heating_rate_c_min", "mass_g", "diameter_mm"):
            bounds = controls.get(key)
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"World needs explicit requested bounds for {key}")
            low, high = (finite(x, f"{key} bound") for x in bounds)
            value = finite(condition.get(key), key)
            if low <= 0 or high < low or not low <= value <= high:
                raise ValueError(f"Unsupported requested {key}: outside declared conditional world")
        temp_bounds = world_config.get("temperature_range_c")
        if not isinstance(temp_bounds, list) or len(temp_bounds) != 2:
            raise ValueError("World needs explicit temperature_range_c")
        low, high = (finite(x, "temperature bound") for x in temp_bounds)
        if not low <= condition["temperature_c"] <= high:
            raise ValueError("Requested temperature outside declared world; no extrapolation")
        cycle_recipe = {
            k: condition[k]
            for k in ("temperature_c", "pressure_mpa", "hold_time_s", "heating_rate_c_min")
        }
        if cycle in cycle_recipes and cycle_recipes[cycle] != cycle_recipe:
            raise ValueError("One cycle cannot bind incompatible requested SPS conditions")
        cycle_recipes[cycle] = cycle_recipe
        cycle_rng = _rng(seed, "cycle:" + cycle)
        prep_rng = _rng(seed, "preparation:" + prep)
        measure_rng = _rng(seed, "measurement:" + sid)
        condition["temperature_c"] += finite(
            world_config.get("temperature_bias_c", 0.0), "temperature_bias_c"
        )
        condition["temperature_c"] += float(
            cycle_rng.normal(0, _sd(world_config, "temperature_readback_sd_c"))
        )
        if not low <= condition["temperature_c"] <= high:
            raise ValueError("Delivered temperature outside declared world; no extrapolation")
        condition["pressure_mpa"] += float(
            cycle_rng.normal(0, _sd(world_config, "pressure_readback_sd_mpa"))
        )
        if condition["pressure_mpa"] <= 0:
            raise ValueError("Synthetic delivered pressure is nonpositive")
        density = _curve(world_config, "density_curve", condition["temperature_c"])
        for variable, ref, coeff in [
            (
                "pressure_mpa",
                "reference_pressure_mpa",
                "density_pressure_sensitivity_g_cm3_per_mpa",
            ),
            ("hold_time_s", "reference_hold_time_s", "density_hold_sensitivity_g_cm3_per_s"),
            (
                "heating_rate_c_min",
                "reference_heating_rate_c_min",
                "density_ramp_sensitivity_g_cm3_per_c_min",
            ),
        ]:
            density += (condition[variable] - finite(world_config[ref], ref)) * finite(
                world_config.get(coeff, 0.0), coeff
            )
        density += float(prep_rng.normal(0, _sd(world_config, "density_preparation_sd_g_cm3")))
        density += float(cycle_rng.normal(0, _sd(world_config, "density_cycle_sd_g_cm3")))
        if not 0 < density <= finite(world_config["reference_density_g_cm3"], "reference_density"):
            raise ValueError("Constructed true density outside physical bounds; revise world")
        delivered[sid] = condition
        selection = specimen.get(
            "measurements", request.get("measurements", ["density_g_cm3", "sps_log"])
        )
        if not isinstance(selection, list) or not all(isinstance(x, str) for x in selection):
            raise ValueError("measurements must be a list of quantity names")
        descriptor_quantities = {"grain_d90_nm", "coarse_number_fraction", "porosity_area_fraction"}
        unknown = set(selection) - ({"density_g_cm3", "grain_size_nm", "sps_log", "microstructure"}
                                    | descriptor_quantities)
        if unknown:
            raise ValueError(f"Unsupported synthetic measurements: {sorted(unknown)}")
        base = {
            "specimen_id": sid,
            "preparation_id": prep,
            "cycle_id": cycle,
            "lot_id": lot,
            "sample_kind": "specimen",
            "origin": "synthetic",
            "status": "conditional_synthetic",
            "noise_status": "declared_not_calibrated",
            "process_stage": "initial_sps",
        }
        extended = "microstructure" in world_config
        micro = None
        if extended and (({"grain_size_nm", "microstructure"} | descriptor_quantities) & set(selection)):
            micro = simulate_microstructure(
                condition, world_config, _rng(seed, "microstructure:" + sid)
            )
        if "density_g_cm3" in selection:
            sd = _sd(world_config, "density_measurement_sd_g_cm3")
            observed = density + float(measure_rng.normal(0, sd))
            observations.append(
                {
                    **base,
                    "observation_id": sid + ":density",
                    "quantity": "density_g_cm3",
                    "value": observed,
                    "unit": "g/cm^3",
                    "uncertainty": {"sd": sd, "kind": "synthetic_measurement_only"},
                    "method": "synthetic_density_observation",
                    **({"statistical_unit": "specimen", "measurement_population": "bulk_specimen",
                        "measurement_statistic": "density"} if extended else {}),
                }
            )
        if "sps_log" in selection:
            observations.append(
                {
                    **base,
                    "observation_id": sid + ":sps_log",
                    "quantity": "sps_log",
                    "value": _sps_log(condition, world_config, density),
                    "unit": "mixed",
                    "method": "synthetic_process_readback",
                }
            )
        if "grain_size_nm" in selection:
            resolved = world_config.get("grain_boundaries_resolved", False)
            if not isinstance(resolved, bool):
                raise ValueError("grain_boundaries_resolved must be boolean")
            sd = _sd(world_config, "grain_measurement_sd_nm")
            grain = None
            if micro is not None:
                grain = micro["grain_distribution"]["mean_nm"] if resolved else None
                sd = None  # Feature dispersion is not measurement or predictive SD.
            elif resolved:
                grain = _curve(world_config, "grain_curve", condition["temperature_c"])
                grain += float(prep_rng.normal(0, _sd(world_config, "grain_preparation_sd_nm")))
                grain += float(cycle_rng.normal(0, _sd(world_config, "grain_cycle_sd_nm")))
                grain += float(measure_rng.normal(0, sd))
                if grain <= 0:
                    raise ValueError("Nonpositive synthetic grain observation")
            observations.append(
                {
                    **base,
                    "observation_id": sid + ":grain",
                    "quantity": "grain_size_nm",
                    "value": grain,
                    "unit": "nm",
                    "resolved_grain_boundaries": resolved,
                    "status": "conditional_synthetic" if resolved else "unresolved",
                    "uncertainty": {
                        "sd": sd if resolved else None,
                        "kind": "synthetic_measurement_only",
                    },
                    "method": "synthetic_microstructure_descriptor_not_an_image",
                    "scope": "selected specimen descriptor; no actual microscopy",
                    "measurement_population": "synthetic_descriptor",
                    "measurement_statistic": "constructed_scalar_descriptor",
                    **({"statistical_unit": "specimen",
                        "measurement_population": "selected_sections",
                        "measurement_statistic": "number_mean",
                        "method": "constructed_features_not_microscopy_images"} if extended else {}),
                }
            )
        if "microstructure" in selection:
            if micro is None:
                raise ValueError("microstructure requires an explicitly extended world")
            observations.append({**base, "observation_id": sid + ":microstructure",
                                 "quantity": "microstructure", "value": micro, "unit": "mixed",
                                 "method": "constructed_features_not_microscopy_images",
                                 "statistical_unit": "specimen", "measurement_population": "selected_sections"})
        for quantity in sorted(descriptor_quantities & set(selection)):
            if micro is None:
                raise ValueError("microstructure descriptors require an explicitly extended world")
            distribution = micro["grain_distribution"]
            key = {"grain_d90_nm": "d90_nm", "coarse_number_fraction": "coarse_number_fraction",
                   "porosity_area_fraction": "porosity_area_fraction"}[quantity]
            value = micro[key] if quantity == "porosity_area_fraction" else (
                distribution[key] if distribution is not None else None)
            observations.append({**base, "observation_id": sid + ":" + quantity,
                "quantity": quantity, "value": value,
                "unit": "nm" if quantity == "grain_d90_nm" else "1",
                "status": "conditional_synthetic" if value is not None else "unresolved",
                "method": "constructed_features_not_microscopy_images",
                "measurement_population": "selected_sections",
                "measurement_statistic": "area_fraction" if quantity == "porosity_area_fraction"
                else "d90" if quantity == "grain_d90_nm" else "coarse_number_fraction",
                "coarse_threshold_nm": world_config["microstructure"]["coarse_threshold_nm"],
                "resolved_grain_boundaries": micro["resolved_grain_boundaries"],
                "statistical_unit": "specimen", "uncertainty": None,
                "scope": "selected constructed sections; not independent bulk or magnetic evidence"})
    cycle_duration = {}
    initial = finite(world_config.get("initial_temperature_c", 25.0), "initial_temperature_c")
    cooling = finite(world_config.get("cooling_rate_c_min", 100.0), "cooling_rate_c_min")
    if cooling <= 0:
        raise ValueError("Positive cooling rate required for modeled cycle duration")
    for actual in delivered.values():
        delta = actual["temperature_c"] - initial
        if delta <= 0:
            raise ValueError("Delivered peak must exceed initial temperature")
        cycle_duration[actual["cycle_id"]] = (
            delta * 60.0 / actual["heating_rate_c_min"]
            + actual["hold_time_s"]
            + delta * 60.0 / cooling
        )
    payload = {
        # Public identity must not fingerprint private response-world parameters or seed.
        # The owner retains content_hash for private binding; never release it as acquired evidence.
        "delivery_id": "sim-" + hashlib.sha256(_canonical(request)).hexdigest()[:20],
        "origin": "synthetic",
        "observations": observations,
        "delivered_conditions": delivered,
        "manufactured_specimen_ids": list(delivered),
        "simulation_accounting": {
            "status": "conditional_synthetic_not_physical_throughput",
            "input_mass_g": sum(actual["mass_g"] for actual in delivered.values()),
            "modeled_cycle_duration_s": cycle_duration,
            "summed_modeled_cycle_duration_s": sum(cycle_duration.values()),
            "accounting_basis": "Declared ramp/hold/cool trajectories, one count per cycle",
            "excluded": [
                "setup",
                "handling",
                "measurement",
                "queue time",
                "processing losses",
                "accepted output",
                "energy",
                "money",
            ],
        },
        "complete": True,
        "provenance": {
            "evidence_class": "conditional_synthetic",
            "model_version": "conditional-world-v3-distributions" if "microstructure" in world_config
            else "conditional-world-v2-semantic-labels",
            "request_world_seed_sha256": content_hash,
            "physical_execution": False,
        },
    }
    _canonical(payload)  # Fail before returning nonfinite/non-JSON data.
    return payload
