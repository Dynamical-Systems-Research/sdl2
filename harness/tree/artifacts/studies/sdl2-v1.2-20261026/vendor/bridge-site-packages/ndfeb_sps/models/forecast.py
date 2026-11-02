"""Bounded empirical calculations, not general sintering kinetics or RPI calibration.

The source interpolator is deliberately simple. Recipe slices are never pooled.
It cannot identify pressure/time coefficients or invent measurement variance.
"""

import math
from collections import defaultdict

import numpy as np

from .sources import PUBLIC_SOURCE_VERSION, source_metadata, source_observations, source_rows

RECIPE = ("pressure_mpa", "heating_rate_c_min", "hold_time_s", "post_treatment")
GEOMETRY = ("mass_g", "diameter_mm")
PROCESS_CONTEXT = ("process_stage", "die_material")
GRAIN_CONTEXT = ("measurement_population", "measurement_statistic")
UNITS = {"density_g_cm3": "g/cm^3", "grain_size_nm": "nm"}


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004 - public payload contract
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def prediction(mean, unit, status, scope, sd=None, uncertainty_status="not_estimated", **extra):
    return {
        "mean": mean,
        "sd": sd,
        "unit": unit,
        "status": status,
        "scope": scope,
        "uncertainty_status": uncertainty_status,
        **extra,
    }


def _unsupported(quantity, reason):
    return prediction(None, UNITS[quantity], "unsupported", reason)


def _match_recipe(left, right):
    return all(left.get(key) == right.get(key) for key in RECIPE)


def _match_evidence_recipe(actual, condition):
    tolerances = condition.get("evidence_recipe_tolerances", {})
    for key in RECIPE:
        if actual.get(key) is None or condition.get(key) is None:
            return False
        if actual.get(key) == condition.get(key):
            continue
        tolerance = tolerances.get(key)
        if tolerance is None or key == "post_treatment":
            return False
        tolerance = finite(tolerance, "evidence recipe tolerance")
        if tolerance < 0 or actual.get(key) is None or condition.get(key) is None:
            return False
        if abs(finite(actual[key], key) - finite(condition[key], key)) > tolerance:
            return False
    return True


def _source_basis(condition):
    return condition.get("source_basis") or {
        "mqu_f_sps": "mishra_2023",
        "recycled_hddr_sps": "ikram_2019",
    }.get(condition.get("route"))


def _context(condition, quantity):
    keys = ("lot_id", "route", "temperature_c", *RECIPE, *GEOMETRY, *PROCESS_CONTEXT)
    if quantity == "grain_size_nm":
        keys += GRAIN_CONTEXT
    return {
        "requested": {key: condition.get(key) for key in keys},
        "unresolved_request_fields": [key for key in keys if condition.get(key) is None],
        "limits": [
            "Context matching is not physical calibration or proof of causal sufficiency",
            "Powder chemistry, oxidation, preparation and initial structure are not inferred",
            "Missing optional context does not identify a real measurement population",
        ],
    }


def source_applicability(condition: dict, source_id=None, quantity="density_g_cm3") -> dict:
    """Inspect explicit support without treating missing source settings as wildcards.

    Wu observations are queryable facts, never an added response law. Legacy
    density slices retain their numerical definition and conditional lot transfer.
    """
    basis = source_id or _source_basis(condition)
    out = {
        "source_id": basis,
        "source_packet_version": PUBLIC_SOURCE_VERSION,
        "quantity": quantity,
        "prediction_supported": False,
        "context": _context(condition, quantity),
        "source_rows": [],
        "unknown_source_fields": [],
        "mismatches": [],
    }
    if basis == "wu_2025":
        checks = []
        for row in source_observations(basis):
            actual = {**row["conditions"], "process_stage": row["process_stage"]}
            measurement = row["measurements"].get(quantity)
            matched, unknown, different = [], [], []
            keys = set(RECIPE + GEOMETRY + PROCESS_CONTEXT + ("route", "lot_id", "temperature_c"))
            keys.update(key for key in condition if key in actual)
            if quantity == "grain_size_nm" or measurement is not None:
                keys.update(GRAIN_CONTEXT)
                actual.update({key: (measurement or {}).get(key) for key in GRAIN_CONTEXT})
            for key in sorted(keys):
                if condition.get(key) is None:
                    continue
                if actual.get(key) is None:
                    unknown.append(key)
                elif actual[key] != condition[key]:
                    different.append(key)
                else:
                    matched.append(key)
            checks.append(
                {
                    "observation_id": row["observation_id"],
                    "process_stage": row["process_stage"],
                    "available_quantities": sorted(row["measurements"]),
                    "quantity_present": measurement is not None,
                    "matched_fields": matched,
                    "unknown_source_fields": unknown,
                    "mismatches": different,
                    "conflicted": row["excluded_from_unconflicted_evidence"],
                    "model_fit_eligible": False,
                }
            )
        out.update(
            status="source_observations_only",
            observation_checks=checks,
            reason="No Wu response model: coupled process changes and unquantified dwell",
            unknown_source_fields=[
                "hold_time_s",
                "lot_id",
                "density_reference_g_cm3",
                "independent_manufacture_n",
                "grain_error_bar_definition",
            ],
            limitations=[
                "Relative density percent is not absolute density without its source denominator",
                "Fine-region grain size is not whole-specimen grain size or powder particle size",
                "Post-hot-deformation dimensions and magnetics cannot calibrate initial SPS",
                "A steady-displacement endpoint is not a fixed-duration hold",
            ],
        )
        return out
    if quantity != "density_g_cm3":
        out.update(status="unsupported", reason="No source-fitted model for this quantity")
        return out
    rows = [
        r
        for r in source_rows(basis)
        if basis is not None
        and r["default_fit"]
        and r["route"] == condition.get("route")
        and _match_recipe(condition, r)
    ]
    if not rows:
        out.update(status="unsupported", reason="No unconflicted source recipe slice matches")
        return out
    rows.sort(key=lambda r: r["temperature_c"])
    expected_stage = "initial_sps" if rows[0]["post_treatment"] == "none" else "post_sinter_anneal"
    # Do not invent a tooling identity for legacy rows whose numeric table lacks it.
    source_context = {"process_stage": expected_stage, "die_material": None}
    out["source_context"] = source_context
    out["unknown_source_fields"] = [
        "die_material",
        "actual_temperature_field",
        "physical_repeatability",
    ]
    for key in PROCESS_CONTEXT:
        if condition.get(key) is not None and source_context[key] != condition[key]:
            out["mismatches"].append(key)
    out["source_rows"] = [r["row_id"] for r in rows]
    out["temperature_range_c"] = [rows[0]["temperature_c"], rows[-1]["temperature_c"]]
    out["source_lot"] = rows[0]["source_lot"]
    out["source_geometry"] = {key: rows[0][key] for key in GEOMETRY}
    transfer = [key for key in GEOMETRY if condition.get(key) != rows[0][key]]
    if condition.get("lot_id") != rows[0]["source_lot"]:
        transfer.insert(0, "lot")
    out["transfer_mismatches"] = transfer
    temp = finite(condition.get("temperature_c"), "temperature_c")
    reason = None
    if out["mismatches"]:
        reason = "Requested process context is different or unknown in source: " + ", ".join(
            out["mismatches"]
        )
    elif not rows[0]["temperature_c"] <= temp <= rows[-1]["temperature_c"]:
        reason = "Temperature outside source slice; no extrapolation"
    elif transfer and not condition.get("allow_source_transfer", False):
        reason = "Source transfer needs explicit acknowledgment: " + ", ".join(transfer)
    out.update(
        status="unsupported"
        if reason
        else "conditional_source_transfer"
        if transfer
        else "source_scoped_interpolation",
        prediction_supported=reason is None,
        reason=reason or "Legacy recipe-slice interpolation; unmeasured context remains unresolved",
    )
    return out


def _source_density(condition):
    route = condition.get("route")
    basis = _source_basis(condition)
    applicability = source_applicability(condition, basis)
    if not applicability["prediction_supported"]:
        return {
            **_unsupported("density_g_cm3", applicability["reason"]),
            "applicability": applicability,
        }, []
    rows = [
        r
        for r in source_rows(basis)
        if r["default_fit"] and r["route"] == route and _match_recipe(condition, r)
    ]
    if not rows:
        return _unsupported("density_g_cm3", "No unconflicted source recipe slice matches"), []
    temp = finite(condition.get("temperature_c"), "temperature_c")
    rows.sort(key=lambda r: r["temperature_c"])
    if not rows[0]["temperature_c"] <= temp <= rows[-1]["temperature_c"]:
        return _unsupported(
            "density_g_cm3", "Temperature outside source slice; no extrapolation"
        ), []
    mismatch = []
    if condition.get("lot_id") != rows[0]["source_lot"]:
        mismatch.append("lot")
    for key in ("mass_g", "diameter_mm"):
        if condition.get(key) != rows[0][key]:
            mismatch.append(key)
    if mismatch and not condition.get("allow_source_transfer", False):
        return _unsupported(
            "density_g_cm3", "Source transfer needs explicit acknowledgment: " + ", ".join(mismatch)
        ), []
    value = float(
        np.interp(temp, [r["temperature_c"] for r in rows], [r["density_g_cm3"] for r in rows])
    )
    scope = {
        "source_id": basis,
        "source_lot": rows[0]["source_lot"],
        "route": route,
        "temperature_range_c": [rows[0]["temperature_c"], rows[-1]["temperature_c"]],
        "recipe": {k: condition[k] for k in RECIPE},
        "transfer_mismatches": mismatch,
    }
    status = "conditional_source_transfer" if mismatch else "source_scoped_interpolation"
    out = prediction(
        value,
        "g/cm^3",
        status,
        scope,
        method="piecewise_linear_temperature_slice",
        source_rows=[r["row_id"] for r in rows],
        applicability=applicability,
    )
    return out, source_metadata()[basis]["warnings"]


def _eligible_evidence(condition, evidence, quantity):
    accepted, rejected = [], []
    seen = set()
    for row in evidence or []:
        if row.get("quantity") != quantity or row.get("superseded", False):
            continue
        key = row.get("evidence_id", row.get("observation_id"))
        if key is None or key in seen:
            rejected.append("Observation lacks unique identity or is duplicated")
            continue
        seen.add(key)
        if row.get("status") in {"unresolved", "invalid", "missing", "rejected", "unavailable"}:
            continue
        if row.get("origin") not in {"synthetic", "physical"}:
            rejected.append("Unlabeled evidence excluded")
            continue
        if row.get("unit") != UNITS[quantity] or row.get("value") is None:
            continue
        actual = row.get("delivered_conditions") or {}
        # Actual delivered conditions, never requested conditions substituted for missing readbacks.
        if (actual.get("lot_id") or row.get("lot_id")) != condition.get("lot_id"):
            rejected.append("Different or missing lot excluded from updating")
            continue
        material_route = actual.get("route") or row.get("material_route")
        if material_route != condition.get("route"):
            rejected.append("Different or missing delivered route excluded from updating")
            continue
        if not _match_evidence_recipe(actual, condition):
            rejected.append("Delivered recipe outside declared comparability tolerance")
            continue
        if any(
            actual.get(k) is None or condition.get(k) is None or actual[k] != condition[k]
            for k in GEOMETRY
        ):
            rejected.append("Different or missing actual mass/diameter excluded from updating")
            continue
        context_keys = PROCESS_CONTEXT + (GRAIN_CONTEXT if quantity == "grain_size_nm" else ())
        incompatible = [
            key
            for key in context_keys
            if condition.get(key) is not None
            and (actual.get(key) if actual.get(key) is not None else row.get(key)) != condition[key]
        ]
        if incompatible:
            rejected.append(
                "Different or unknown observation context excluded: " + ", ".join(incompatible)
            )
            continue
        if actual.get("temperature_c") is None:
            rejected.append("Missing delivered temperature excluded from updating")
            continue
        if quantity == "grain_size_nm" and not row.get("resolved_grain_boundaries", False):
            rejected.append("Unresolved microscopy excluded from grain updating")
            continue
        temp = finite(actual["temperature_c"], "delivered temperature")
        value = finite(row["value"], quantity)
        if value <= 0:
            raise ValueError(f"{quantity} evidence must be positive")
        accepted.append((temp, value, row))
    origins = {r[2]["origin"] for r in accepted}
    selected_origin = condition.get("evidence_origin")
    if selected_origin:
        accepted = [r for r in accepted if r[2]["origin"] == selected_origin]
    elif len(origins) > 1:
        rejected.append("Synthetic and physical evidence not pooled; choose evidence_origin")
        accepted = []
    context_keys = PROCESS_CONTEXT + (GRAIN_CONTEXT if quantity == "grain_size_nm" else ())
    for key in context_keys:
        known = {
            _observation_context(row, key)
            for _, _, row in accepted
            if _observation_context(row, key) is not None
        }
        if len(known) > 1:
            rejected.append("Different known observation contexts not pooled: " + key)
            accepted = []
    return accepted, rejected


def _observation_context(row, key):
    actual = row.get("delivered_conditions") or {}
    return actual.get(key) if actual.get(key) is not None else row.get(key)


def _empirical_baseline(
    condition: dict, evidence: list | None = None, quantity: str = "density_g_cm3"
) -> dict:
    """Same-evidence comparator with explicitly selected local response assumptions.

    Fields/sections within one manufactured specimen are first collapsed. Measured
    repeat scatter is only estimated from independent preparation+cycle clusters.
    Default cell interpolation is exact-recipe and does not smooth readback jitter.
    Opt-in local_linear_actual_temperature fits delivered T, never requested T;
    its declared local-linearity/nuisance assumptions must be stated by the caller.
    """
    if quantity not in UNITS:
        raise ValueError("Unsupported baseline quantity")
    accepted, warnings = _eligible_evidence(condition, evidence, quantity)
    if not accepted:
        return {
            **_unsupported(quantity, "No comparable released observations"),
            "warnings": warnings,
        }
    # A preparation or cycle shared between specimens can induce correlation.
    # Only observations with both independent identifiers support repeat variance.
    specimen_groups = defaultdict(list)
    for t, value, row in accepted:
        identity = row.get("specimen_id") or row.get("evidence_id") or row["observation_id"]
        specimen_groups[(t, identity)].append((value, row))
    cells = defaultdict(list)
    for (temp, _), members in specimen_groups.items():
        row = members[0][1]
        actual = row.get("delivered_conditions", {})
        prep = row.get("preparation_id", actual.get("preparation_id"))
        cycle = row.get("cycle_id", actual.get("cycle_id"))
        cells[temp].append((float(np.mean([x[0] for x in members])), prep, cycle))
    temperatures = sorted(cells)
    target = finite(condition.get("temperature_c"), "temperature_c")
    if not temperatures[0] <= target <= temperatures[-1]:
        return {
            **_unsupported(quantity, "No baseline extrapolation beyond local evidence"),
            "warnings": warnings,
        }
    method = condition.get("empirical_method", "specimen_cell_mean_interpolation")
    if method == "local_linear_actual_temperature":
        return _linear_baseline(condition, accepted, cells, quantity, target, warnings)
    if method != "specimen_cell_mean_interpolation":
        raise ValueError("Unknown empirical_method")
    means = [float(np.mean([x[0] for x in cells[t]])) for t in temperatures]
    mean = float(np.interp(target, temperatures, means))
    variance_numerator, variance_df = 0.0, 0
    for t, members in cells.items():
        preps = [x[1] for x in members]
        cycles = [x[2] for x in members]
        if (
            len(members) > 1
            and None not in preps
            and None not in cycles
            and len(set(preps)) == len(preps)
            and len(set(cycles)) == len(cycles)
        ):
            vals = np.array([x[0] for x in members])
            variance_numerator += float(np.sum((vals - vals.mean()) ** 2))
            variance_df += len(vals) - 1
    sd = math.sqrt(variance_numerator / variance_df) if variance_df else None
    # This is observed repeat scatter, not a calibrated new-lot predictive interval.
    return prediction(
        mean,
        UNITS[quantity],
        "local_empirical_update",
        {
            "lot_id": condition.get("lot_id"),
            "route": condition.get("route"),
            "temperature_range_c": [temperatures[0], temperatures[-1]],
            "evidence_origin": accepted[0][2]["origin"],
            "conditional_recipe_tolerances": condition.get("evidence_recipe_tolerances", {}),
        },
        sd=sd,
        uncertainty_status="observed_repeat_scatter_only" if sd is not None else "not_estimated",
        method="specimen_cell_mean_interpolation",
        evidence_ids=[r[2].get("evidence_id", r[2].get("observation_id")) for r in accepted],
        independent_repeat_df=variance_df,
        warnings=warnings,
        limitations=[
            "Not a confidence interval or calibrated predictive interval",
            "Between-batch discrepancy and interpolation uncertainty unestimated",
        ],
    )


def empirical_baseline(
    condition: dict, evidence: list | None = None, quantity: str = "density_g_cm3"
) -> dict:
    """Conventional comparator with complete requested-context and exclusion labels."""
    out = _empirical_baseline(condition, evidence, quantity)
    out["applicability"] = _context(condition, quantity)
    out["applicability"]["evidence_scope"] = out["scope"]
    out["applicability"]["numeric_prediction_supported"] = out["mean"] is not None
    if out["mean"] is not None:
        accepted, _ = _eligible_evidence(condition, evidence, quantity)
        keys = PROCESS_CONTEXT + (GRAIN_CONTEXT if quantity == "grain_size_nm" else ())
        out["applicability"]["observed_context"] = {
            key: sorted(
                {
                    _observation_context(row, key)
                    for _, _, row in accepted
                    if _observation_context(row, key) is not None
                }
            )
            for key in keys
        }
        out["applicability"]["unknown_observation_context_fields"] = [
            key
            for key in keys
            if any(_observation_context(row, key) is None for _, _, row in accepted)
        ]
    return out


def _linear_baseline(condition, accepted, cells, quantity, target, warnings):
    rationale = condition.get("empirical_assumption_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Local linear fitting needs empirical_assumption_rationale")
    points = [
        (t, value, prep, cycle) for t, members in cells.items() for value, prep, cycle in members
    ]
    if len({p[0] for p in points}) < 2:
        return {
            **_unsupported(quantity, "At least two distinct delivered temperatures needed"),
            "warnings": warnings,
        }
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    design = np.column_stack((np.ones(len(points)), x - target))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        return {
            **_unsupported(quantity, "Actual temperature design is rank deficient"),
            "warnings": warnings,
        }
    mean = float(coefficients[0])
    if mean <= 0:
        return {
            **_unsupported(quantity, "Local linear fit gives a nonphysical prediction"),
            "warnings": warnings,
        }
    preps, cycles = [p[2] for p in points], [p[3] for p in points]
    independent = (
        None not in preps
        and None not in cycles
        and len(set(preps)) == len(points)
        and len(set(cycles)) == len(points)
    )
    df = len(points) - 2 if independent and len(points) > 2 else 0
    residual = y - design @ coefficients
    sd = math.sqrt(float(residual @ residual) / df) if df else None
    return prediction(
        mean,
        UNITS[quantity],
        "local_empirical_update",
        {
            "lot_id": condition.get("lot_id"),
            "route": condition.get("route"),
            "temperature_range_c": [float(x.min()), float(x.max())],
            "evidence_origin": accepted[0][2]["origin"],
            "conditional_recipe_tolerances": condition.get("evidence_recipe_tolerances", {}),
            "assumption_rationale": rationale,
            "actual_recipe_ranges": {
                key: [
                    min(r[2]["delivered_conditions"][key] for r in accepted),
                    max(r[2]["delivered_conditions"][key] for r in accepted),
                ]
                for key in RECIPE
                if key != "post_treatment"
            },
        },
        sd=sd,
        uncertainty_status="observed_residual_scatter_only" if sd is not None else "not_estimated",
        method="local_linear_actual_temperature",
        evidence_ids=[r[2].get("evidence_id", r[2].get("observation_id")) for r in accepted],
        independent_repeat_df=df,
        n_specimens=len(points),
        slope_per_c=float(coefficients[1]),
        warnings=warnings,
        limitations=[
            "Linear mean within observed actual temperature range is a declared assumption",
            "Nuisance settings within explicit tolerance are treated as locally comparable",
            "Residual scatter is not a confidence or calibrated predictive interval",
            "Residuals confound curvature, measurement and preparation/cycle variation",
            "Between-batch and new-lot discrepancy remain unestimated",
        ],
    )


def _grain(condition):
    assumption = condition.get("grain_assumption")
    if assumption:
        if not assumption.get("rationale"):
            raise ValueError("grain_assumption requires rationale")
        points = assumption.get("temperature_mean_pairs", [])
        if len(points) < 2:
            raise ValueError("grain_assumption needs at least two explicit points")
        points = sorted((finite(x, "grain T"), finite(y, "grain size")) for x, y in points)
        if any(y <= 0 for _, y in points) or len({x for x, _ in points}) != len(points):
            raise ValueError("Grain points need positive sizes and distinct temperatures")
        temp = finite(condition.get("temperature_c"), "temperature_c")
        if not points[0][0] <= temp <= points[-1][0]:
            return _unsupported("grain_size_nm", "Outside declared grain assumption")
        return prediction(
            float(np.interp(temp, *zip(*points, strict=True))),
            "nm",
            "conditional_assumption",
            assumption["rationale"],
            method="declared_curve_interpolation",
        )
    return _unsupported(
        "grain_size_nm",
        "No portable grain kinetics or resolved local evidence; "
        "source grain distributions are retained as observations, not predictive SD",
    )


def forecast(condition: dict, evidence: list | None = None) -> dict:
    """Predict only declared source support or explicit assumptions.

    Actual RPI lot/recipe are not defaulted. Local updates use only released,
    nonsuperseded evidence with actual delivered conditions and compatible history.
    """
    if not isinstance(condition, dict):
        raise ValueError("condition must be a dictionary")  # noqa: TRY004 - public payload contract
    density, warnings = _source_density(condition)
    grain = _grain(condition)
    for quantity, current in (("density_g_cm3", density), ("grain_size_nm", grain)):
        local = empirical_baseline(condition, evidence, quantity)
        warnings.extend(local.get("warnings", []))
        if local["mean"] is not None:
            current.clear()
            current.update(local)
        else:
            current["local_evidence_check"] = {
                "status": local["status"],
                "reason": local["scope"],
                "warnings": local.get("warnings", []),
                "applicability": local["applicability"],
            }
        current.setdefault("applicability", _context(condition, quantity))
        sd_key = {"density_g_cm3": "density_sd_g_cm3", "grain_size_nm": "grain_sd_nm"}[quantity]
        assumed = condition.get("uncertainty", {}).get(sd_key)
        if assumed is not None and current["mean"] is not None:
            sd = finite(assumed, sd_key)
            if sd < 0:
                raise ValueError("Assumed SD must be nonnegative")
            current["sd"] = sd
            current["uncertainty_status"] = "explicit_conditional_assumption"
    return {
        "density_g_cm3": density,
        "grain_size_nm": grain,
        "provenance": "public_source_or_labeled_local_evidence; not RPI validation",
        "applicability": {
            "density_g_cm3": density["applicability"],
            "grain_size_nm": grain["applicability"],
            "magnetic_performance": "unpredicted",
        },
        "warnings": sorted(set(warnings)),
        "model_version": "source-scoped-v2",
        "source_packet_version": PUBLIC_SOURCE_VERSION,
    }


def source_replay_report() -> dict:
    """Deterministic source reconstruction and internal interpolation diagnostics.

    Leave-one-temperature-out is an internal diagnostic only. Endpoints are omitted
    because interpolation cannot predict them without unsupported extrapolation.
    Keszler remains an unmatched/censored external study, not scored as a holdout.
    """
    groups = defaultdict(list)
    for row in source_rows():
        if row["default_fit"]:
            key = (row["source_id"], *(row[k] for k in RECIPE))
            groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["temperature_c"])
        x = np.array([r["temperature_c"] for r in rows])
        y = np.array([r["density_g_cm3"] for r in rows])
        errors, null_errors = [], []
        for i in range(1, len(rows) - 1):
            xp, yp = np.delete(x, i), np.delete(y, i)
            errors.append(float(np.interp(x[i], xp, yp) - y[i]))
            null_errors.append(float(np.mean(yp) - y[i]))
        replay_errors = []
        for row in rows:
            condition = {**row, "lot_id": row["source_lot"], "source_basis": row["source_id"]}
            predicted, _ = _source_density(condition)
            replay_errors.append(abs(predicted["mean"] - row["density_g_cm3"]))
        result.append(
            {
                "source_id": key[0],
                "recipe": dict(zip(RECIPE, key[1:], strict=True)),
                "n_source_rows": len(rows),
                "source_replay_max_abs_error_g_cm3": max(replay_errors),
                "interior_loo_n": len(errors),
                "interior_loo_mae_g_cm3": float(np.mean(np.abs(errors))),
                "constant_mean_loo_mae_g_cm3": float(np.mean(np.abs(null_errors))),
            }
        )
    return {
        "status": "source_reconstruction_and_internal_diagnostic_only",
        "slices": result,
        "independent_validation": "none; Keszler settings differ and observations are censored",
        "excluded_rows": [r["row_id"] for r in source_rows() if not r["default_fit"]],
        "physical_noise_fit": "none",
    }
