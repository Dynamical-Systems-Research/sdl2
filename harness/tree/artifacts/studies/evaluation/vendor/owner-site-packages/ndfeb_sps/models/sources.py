"""Read factual source observations; censored and conflicted rows stay identifiable."""

import ast
import csv
import json
from importlib.resources import files

PUBLIC_SOURCE_VERSION = "ndfeb-public-sources-v2"

NUMERIC = {
    "temperature_c",
    "pressure_mpa",
    "heating_rate_c_min",
    "hold_time_s",
    "mass_g",
    "diameter_mm",
    "density_g_cm3",
    "relative_density_pct",
    "lower_g_cm3",
    "upper_g_cm3",
}


def source_rows(source_id: str | None = None) -> list[dict]:
    with files(__package__).joinpath("data/density.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in NUMERIC:
            row[key] = float(row[key]) if row[key] else None
        row["default_fit"] = row["default_fit"] == "true"
    return [row for row in rows if source_id is None or row["source_id"] == source_id]


def source_metadata() -> dict:
    legacy = json.loads(files(__package__).joinpath("data/sources.json").read_text())
    return {**legacy, "wu_2025": _wu_source()["metadata"]}


def _wu_source() -> dict:
    return json.loads(files(__package__).joinpath("data/wu2025.json").read_text())


def source_observations(source_id=None, stage=None, *, include_conflicted=True) -> list[dict]:
    """Versioned source facts, not predictions or additional legacy fit rows.

    Stage and measurement population are intrinsic to each observation. Unknown
    hold time remains null, and reported dimensional spreads are not predictive SD.
    """
    rows = _wu_source()["observations"]
    return [
        row
        for row in rows
        if (source_id is None or row["source_id"] == source_id)
        and (stage is None or row["process_stage"] == stage)
        and (include_conflicted or not row["excluded_from_unconflicted_evidence"])
    ]


def public_source_packet() -> dict:
    """Public facts and limits only; no world config, seeds, or unreleased evidence."""
    wu = _wu_source()
    return {
        "schema_version": 2,
        "source_packet_version": PUBLIC_SOURCE_VERSION,
        "evidence_class": "published_source_facts",
        "sources": source_metadata(),
        "density_rows": source_rows(),
        "observations": source_observations(),
        "source_details": {
            "wu_2025": {
                key: wu[key]
                for key in (
                    "starting_material",
                    "methods",
                    "coarse_region_observations",
                    "thermal_coefficients",
                    "reported_conflicts",
                    "limitations",
                )
            }
        },
        "limitations": [
            "Not RPI calibration or physical campaign evidence",
            "Censored/conflicted source rows retain their status",
            "No inferred physical repeatability from digitization",
            "Powder particle size is not grain size",
            "HDDR and melt-spun routes must remain separate",
            "Wu initial SPS, hot deformation and thermal measurements are distinct stages",
            "Unknown source dwell is not a wildcard for a requested hold time",
        ],
    }


def bootstrap_model_source(*, assimilate_evidence: bool = False) -> str:
    """Standalone owner-supplied starting model for an isolated NumPy interpreter.

    Default is explicitly source-only, not an automatically updating investigator.
    The conventional comparator is separately available with assimilation enabled.
    This packages only the public forecaster and public numerical facts. It does
    not import the simulation module or expose any conditional-world parameter.
    """
    if not isinstance(assimilate_evidence, bool):
        raise ValueError("assimilate_evidence must be a boolean")  # noqa: TRY004
    packet = public_source_packet()
    origin = (
        "owner_supplied_conventional_assimilator"
        if assimilate_evidence
        else "owner_supplied_source_only_prior"
    )
    header = (
        f"# {origin}; not an agent-discovered model.\n"
        "import copy\nimport json\n"
        f"_PUBLIC = json.loads({json.dumps(packet, sort_keys=True)!r})\n"
        "PUBLIC_SOURCE_VERSION = _PUBLIC['source_packet_version']\n"
        "def source_rows(source_id=None):\n"
        "    return [copy.deepcopy(r) for r in _PUBLIC['density_rows'] "
        "if source_id is None or r['source_id'] == source_id]\n"
        "def source_metadata():\n    return copy.deepcopy(_PUBLIC['sources'])\n"
        "def source_observations(source_id=None, stage=None, *, include_conflicted=True):\n"
        "    return [copy.deepcopy(r) for r in _PUBLIC['observations'] "
        "if (source_id is None or r['source_id'] == source_id) "
        "and (stage is None or r['process_stage'] == stage) "
        "and (include_conflicted or not r['excluded_from_unconflicted_evidence'])]\n"
    )
    body = files(__package__).joinpath("forecast.py").read_text()
    lines = body.splitlines(keepends=True)
    imports = [
        node
        for node in ast.parse(body).body
        if isinstance(node, ast.ImportFrom) and node.module == "sources" and node.level
    ]
    for node in reversed(imports):
        del lines[node.lineno - 1 : node.end_lineno]
    body = "".join(lines)
    body = body.replace("def forecast(condition:", "def _condition_forecast(condition:")
    footer = """
def forecast(payload):
    request = payload["request"]
    evidence = __EVIDENCE_EXPRESSION__
    predictions = []
    for specimen in request["specimens"]:
        condition = dict(specimen["conditions"])
        condition.setdefault("lot_id", request["lot_id"])
        out = _condition_forecast(condition, evidence)
        for quantity in ("density_g_cm3", "grain_size_nm"):
            if out[quantity]["mean"] is None:
                out[quantity]["reason"] = out[quantity]["scope"]
        predictions.append({"specimen_id": specimen["specimen_id"], **out})
    return {"predictions": predictions, "model_origin": "__MODEL_ORIGIN__"}

if __name__ == "__main__":
    import sys
    print(json.dumps(forecast(json.load(sys.stdin)), allow_nan=False))
"""
    footer = footer.replace(
        "__EVIDENCE_EXPRESSION__",
        'payload.get("evidence", [])'
        if assimilate_evidence
        else "[]  # Explicit source-only prior",
    )
    footer = footer.replace("__MODEL_ORIGIN__", origin)
    return header + body + footer


def conventional_model_source() -> str:
    """Owner-supplied, same-evidence conventional comparator; not agent learning."""
    return bootstrap_model_source(assimilate_evidence=True)
