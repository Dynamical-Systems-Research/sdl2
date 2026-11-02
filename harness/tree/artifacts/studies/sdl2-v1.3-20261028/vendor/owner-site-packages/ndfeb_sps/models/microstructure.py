"""Feature-level descriptors, not a magnetic-performance or grain-growth law.

Number weighting and the selected 2D field are explicit. A coarse threshold is
an analysis choice, not a universal physical cutoff. No feature count is an
independent specimen count and distribution width is not predictive uncertainty.
"""

import math

import numpy as np


def grain_distribution(sizes_nm, *, coarse_threshold_nm):
    if not isinstance(sizes_nm, list) or not sizes_nm:
        raise ValueError("grain distribution needs measured feature sizes")
    values = [*sizes_nm, coarse_threshold_nm]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("feature sizes and coarse threshold must be positive finite numbers")
    a = np.asarray(sizes_nm, dtype=float)
    return {
        "mean_nm": float(a.mean()),
        "median_nm": float(np.median(a)),
        "d90_nm": float(np.quantile(a, .9, method="linear")),
        "maximum_nm": float(a.max()),
        "feature_sample_sd_nm": float(a.std(ddof=1)) if len(a) > 1 else None,
        "coarse_threshold_nm": float(coarse_threshold_nm),
        "coarse_number_fraction": float(np.mean(a > coarse_threshold_nm)),
        "feature_count": len(a),
        "weighting": "number",
        "quantile_method": "linear",
        "population": "observed_features_in_selected_sections",
        "predictive_sd_nm": None,
        "limits": ["Not bulk or volume weighted", "Not independent specimen repeats",
                   "Coarse threshold is declared, not a magnetic acceptance limit"],
    }


def simulate_microstructure(condition, config, rng):
    """Owner-defined stress test of a misleading mean; no physical parameter fit.

Uses a separate RNG so selecting microscopy cannot change prior density/log
observations. Returns raw constructed features, never fabricated SEM images.
"""
    spec = config.get("microstructure")
    if not isinstance(spec, dict) or not spec.get("assumption_rationale"):
        raise ValueError("microstructure selection needs explicit world assumptions")
    count = spec.get("features_per_field")
    fields = spec.get("field_count")
    if type(count) is not int or not 2 <= count <= 1000:
        raise ValueError("features_per_field must be 2..1000")
    if type(fields) is not int or not 1 <= fields <= 20:
        raise ValueError("field_count must be 1..20")

    def curve(name, *, fraction=False):
        pairs = spec.get(name)
        if not isinstance(pairs, list) or len(pairs) < 2:
            raise ValueError(f"{name} requires a declared curve")
        if any(not isinstance(p, list) or len(p) != 2 for p in pairs):
            raise ValueError(f"{name} needs temperature/value pairs")
        if any(type(x) not in (int, float) or not math.isfinite(x) for p in pairs for x in p):
            raise ValueError(f"{name} must be finite")
        pairs = sorted(pairs)
        if len({p[0] for p in pairs}) != len(pairs):
            raise ValueError(f"{name} duplicate temperatures")
        if any(not 0 <= p[1] <= 1 if fraction else p[1] <= 0 for p in pairs):
            raise ValueError(f"{name} invalid values")
        t = condition["temperature_c"]
        if not pairs[0][0] <= t <= pairs[-1][0]:
            raise ValueError(f"{name} outside declared domain")
        return float(np.interp(t, *zip(*pairs, strict=True)))

    fine, coarse = curve("fine_size_curve_nm"), curve("coarse_size_curve_nm")
    fraction = curve("coarse_component_weight_curve", fraction=True)
    porosity = curve("porosity_area_fraction_curve", fraction=True)
    if coarse <= fine:
        raise ValueError("coarse component must exceed fine component")
    cv = spec.get("feature_cv")
    if type(cv) not in (int, float) or not math.isfinite(cv) or not 0 <= cv <= 1:
        raise ValueError("feature_cv must be a declared finite 0..1 assumption")
    sigma = math.sqrt(math.log1p(cv**2))
    concentration = spec.get("porosity_field_concentration")
    if type(concentration) not in (int, float) or not math.isfinite(concentration) or concentration <= 0:
        raise ValueError("porosity_field_concentration must be a positive declared assumption")
    resolved = config.get("grain_boundaries_resolved") is True
    field_records, sizes = [], []
    for i in range(fields):
        features = []
        if resolved:
            means = np.where(rng.random(count) < fraction, coarse, fine)
            features = rng.lognormal(np.log(means) - sigma**2 / 2, sigma).tolist()
            sizes.extend(features)
        field_records.append({"field_id": f"field-{i + 1}", "section_id": "section-1",
                              "grain_sizes_nm": features,
                              "porosity_area_fraction": float(rng.beta(porosity * concentration,
                                  (1 - porosity) * concentration)) if 0 < porosity < 1 else porosity})
    return {
        "fields": field_records,
        "grain_distribution": grain_distribution(
            sizes, coarse_threshold_nm=spec["coarse_threshold_nm"]
        ) if resolved else None,
        "resolved_grain_boundaries": resolved,
        "porosity_area_fraction": float(np.mean([f["porosity_area_fraction"] for f in field_records])),
        "boundary_chemistry": {"status": "unresolved", "value": None,
                               "reason": "No calibrated boundary-composition measurement model"},
        "texture": {"status": "unmeasured", "value": None},
        "magnetic_performance": {"status": "unpredicted", "value": None},
        "method": "constructed_features_not_microscopy_images",
        "sampling": "predeclared_equal_count_fields; no favorable-field selection",
        "porosity_model": "separate declared section curve and beta field scatter; not derived from density",
        "uncertainty": None,
        "limits": ["Conditional stress test, not RPI calibration",
                   "Descriptor variability is feature/field sampling only; delivery drift is unmodeled",
                   "2D section porosity is not bulk void fraction",
                   "Distribution morphology and chemistry cannot establish magnetic performance"],
    }
