"""Prepare one SDL-2 episode from the shared contract, the clean P7 base ledger and an owner world.

  ../python/bin/python -B prepare.py NAME --panel dev --state dev-01     a development-panel world
  ../python/bin/python -B prepare.py auto --panel train --state train-0001  a training-panel world (same design)
  ../python/bin/python -B prepare.py NAME --seed 7300123                 any generator seed: owner checks only (an
                                                                             unselected world is outside the contract's
                                                                             stated lot distribution)

Workspace (agent-visible): START_HERE.md (= CONTRACT.md), LABORATORY.md, library/ and models/supplied/ (empty mount
points: run.mjs bind-mounts env/library-agent and env/models-agent/supplied read-only), outbox/. INITIAL_EVIDENCE.json and COVERAGE.md are written by run.mjs at
initialization. Owner-only (never mounted): simulation-magnetic/ world and runtime, PREPARATION.json.
Training rollouts and evaluation episodes are prepared by this same function; only the world differs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
BRIDGE = STUDY.parent / "bridge"
LAB = STUDY.parent / "material-lab"
OWNER_ENV = STUDY / "python"
BASE = BRIDGE / "env/base-investigation"   # read-only: sqlite immutable, blobs hard-linked
LOT = "SDL2-LOT"
REFERENCE = {"preparation_id": "REF-P", "specimen_id": "REF-B1", "cycle_id": "REF-C1",
             "preparation": {"method": "vacuum_degassing", "temperature_c": 600, "hold_time_s": 900, "feedstock_d50_um": 4.0},
             "conditions": {"temperature_c": 850, "heating_rate_c_min": 10, "hold_time_s": 300, "pressure_mpa": 60,
                            "pressure_schedule": "on_cooling", "mass_g": 0.9, "diameter_mm": 10,
                            "measurement_context": {"piece_id": "P1", "density_method": "geometric", "field_set_id": "REF-ACQ-1"}},
             "measurements": ["density_g_cm3", "hci_kOe", "br_kGs", "bhmax_MGOe"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def link_tree(src, dst):
    """Hard-link a read-only master tree (no extra disk; files stay 0444)."""
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            os.link(Path(root) / f, dst / rel / f)


def reference_request():
    r = REFERENCE
    return {"origin": "synthetic", "operation": "manufacture", "lot_id": LOT,
            "preparations": [{"preparation_id": r["preparation_id"], "lot_id": LOT, "conditions": r["preparation"]}],
            "specimens": [{"specimen_id": r["specimen_id"], "preparation_id": r["preparation_id"], "cycle_id": r["cycle_id"],
                           "conditions": r["conditions"], "measurements": r["measurements"]}]}


def capabilities(world):
    """The twin's public capability description, corrected for this instrument (agent-visible)."""
    from ndfeb_sps.models import campaign_world as campaign, owner_world as owner_world
    caps = campaign.public_capabilities(world["v12"]["base_world"])
    caps["world_family"] = owner_world.PUBLIC_WORLD
    caps["description"] = (
        "Simulates one virtual powder lot. The hidden world is drawn from the same posterior as the supplied physics "
        "package (its README states the data cutoff), not from its stored "
        "rows, with one of its two magnetic "
        "calculations, the coercivity link or the C5 path, as this lot's magnetic response. About half of the lots in this "
        "program have a declared route meeting the requirement. "
        "Instrument corrections: degassing completeness is not a control; Hci is bounded at 30 kOe; the 900 C "
        "hold acts on coercivity through the hold-scaled grain on both magnetic calculations, as in the package. Returns are "
        "conditional computation, not physical observations, and "
        "agreement with the package is not independent validation of it.")
    drop = ("complete degassing", "degassing_completeness", "Magnetic outputs", "Film magnetization:")
    caps["limits"] = [s for s in caps["limits"] if not any(d in s for d in drop)]
    caps["declared_and_uncalibrated"] = [s.replace("Degassing completeness other than the fitted states, complete degassing, an "
                                                   "insulated current path", "An insulated current path")
                                         for s in caps.get("declared_and_uncalibrated", [])]
    caps["limits"] += [
        "Numeric hypothetical demagnetization-corrected loops at 20 C on isotropic bodies; persistent hidden state; "
        "new reading noise per acquisition (field_set_id); independent 2% log reading errors on Hci and Br; the knee field "
        "(Hk) is read with its own error, including a demagnetization correction fixed per piece",
        "Density readings report each method's nominal reading sd; this lot's actual scatter is not stated",
        "SEM distribution readings (d90, coarse fraction) carry a width read from their own field set; the SPS log's "
        "geometry-derived densities carry a calibration error of about 2%",
        "Ideal own-piece magnetic normalization is assumed, not measured; no numeric density is disclosed by a loop",
        "Degassing runs as method none or vacuum degassing at 600 C for 600-900 s; completeness is not a control",
        "No current-panel magnetic validation; the reading model is declared, not a qualified RPI protocol"]
    controls = caps["controls"]
    controls["lot_id"] = LOT
    env = controls["support_envelope"]
    for k in ("degassing_completeness", "normalization_density_g_cm3"):
        env.pop(k, None)
    env.update(degassing_temperature_c=[600.0, 600.0], degassing_hold_time_s=[600.0, 900.0])
    controls["measurement_context"] = [k for k in controls["measurement_context"] if k != "normalization_density_g_cm3"]
    controls.get("measurement_context_c7", {}).pop("normalization_density_g_cm3", None)
    caps["limits"] = [s.replace(", normalized by the body's own stage density unless measurement_context."
                                "normalization_density_g_cm3 states another", ", normalized by each piece's own stage density "
                                "(ideal normalization; no stated density is accepted)") for s in caps["limits"]]
    caps["controls"]["preparation"] = (
        "Every manufacture request includes request.preparations for each preparation it uses, including a reused one with "
        "identical conditions. conditions: method none, or method vacuum_degassing with temperature_c 600 and hold_time_s "
        "600-900; optional feedstock_d50_um (0.5-8.0 um, default 4.0). Degassing completeness is not accepted.")
    caps["measurement_models"]["hci_kOe, br_kGs, bhmax_MGOe, hci_screen_kOe, demagnetization_curve"] = (
        "persistent hypothetical loop state with new reading noise per acquisition")
    caps["resources"] = "Allowance and prices are in START_HERE.md; nothing is blocked."
    return caps


def world_for(panel, state, seed):
    sys.path.insert(0, str(STUDY / "src"))   # owner-side draws resolve owner/ through src/evaluation_world.py
    import evaluation_world as evaluation_world
    if panel:
        states = json.loads((STUDY / "owner" / f"{panel.upper()}_PANEL.json").read_text())["states"]
        row = next(s for s in states if s["name"] == state)
        return row["world"], {"panel": panel, "state": state, "seed": row["seed"], "class": row["class"]}
    meas = {"density_body_sd_g_cm3": 0.02, "link_model_draws": 4000}
    world = evaluation_world.draw("eval", f"v13-adhoc-{seed}", meas, lot_id=LOT, drawn_at="2026-10-28", world_seed=seed)
    return world, {"panel": None, "state": None, "seed": seed, "class": None}


def prepare(name, panel=None, state=None, seed=None, root=None, contract=None):
    """name "auto" gives an opaque episode name: the workspace path is visible to the agent. contract: a pre-registered
    variant of CONTRACT.md (context/CONTRACT.<variant>.md) for the context pilot; default the base contract."""
    contract_path = Path(contract) if contract else HERE / "CONTRACT.md"
    if name == "auto":
        name = "e-" + secrets.token_hex(5)
    root = Path(root) if root else STUDY / "episodes"
    ep = root / name
    if ep.exists():
        raise SystemExit("episode exists; choose a fresh name")
    world, origin = world_for(panel, state, seed)
    campaign = ep / "campaign"
    campaign.mkdir(parents=True)
    with sqlite3.connect(f"file:{BASE / 'campaign/campaign.sqlite3'}?mode=ro&immutable=1", uri=True) as src, \
            sqlite3.connect(campaign / "campaign.sqlite3") as dst:
        src.backup(dst)
    link_tree(BASE / "campaign/blobs", campaign / "blobs")   # content-addressed; the bridge only adds new blobs
    skip = shutil.ignore_patterns("*-shm", "*-wal")
    shutil.copytree(BASE / "simulation", ep / "simulation", ignore=skip)
    shutil.copytree(BASE / "simulation-magnetic/campaign", ep / "simulation-magnetic/campaign", ignore=skip)
    ws = ep / "pi/workspace"
    for mount in ("library", "models/supplied"):   # run.mjs mounts env/library-agent and models-agent read-only here
        (ws / mount).mkdir(parents=True)
    (ws / "outbox").mkdir()
    shutil.copy2(contract_path, ws / "START_HERE.md")
    shutil.copy2(HERE / "LABORATORY.md", ws / "LABORATORY.md")
    with sqlite3.connect(f"file:{BASE / 'pi/scientific_state/investigation.sqlite3'}?mode=ro&immutable=1", uri=True) as src:
        packet = json.loads(src.execute("SELECT value FROM meta WHERE key='sources'").fetchone()[0])
    contract = contract_path.read_text()
    task = packet["task"]
    for k in ("collaboration", "scope_file", "provenance_file"):
        task.pop(k, None)
    task.update(
        objective="Decide, for one virtual powder lot, whether a declared route meets Hci >= 10 kOe and (BH)max >= 11 MGOe "
                  "for three bodies of a new preparation with probability >= 0.95; nominate it or restrict.",
        mission=contract, material_question=contract.split("\n\n")[1],
        scientific_scope="SDL-2: one virtual lot; physical record through P7 as calibration context",
        current_campaign_stage="Physical record through P7; one virtual evaluation lot with one reference body",
        execution_status="Virtual lot only; no physical execution",
        final_record="decision.json with one claim (nominate, restrict or none), premises, next_action and rationale; "
                     "then record_decision stop",
        measurement_capabilities=sorted(set(task.get("measurement_capabilities", []))
                                        | {"hci_kOe", "br_kGs", "bhmax_MGOe", "demagnetization_curve", "hci_screen_kOe"}),
        measurement_limits="See START_HERE.md and LABORATORY.md; the physical record is in library/ and read_evidence.",
        baseline_id="sdl2")
    task["available_resources"] = {"allowance_credits": 570, "new_bodies": 12, "physical_execution": False,
                                   "manufacture_authority": "virtual lot only"}
    task.pop("inventory_label", None)
    task["semantics"].update(
        anneal="Use operation=anneal on an existing virtual body, with input_stage its current stage and an explicit "
               "output_stage. The bridge supplies the executed history: do not write prior_processing_conditions. "
               "anneal_steps follow LABORATORY.md.",
        measure="Use operation=measure on an existing virtual body with input_stage = output_stage = its current stage. "
                "The bridge supplies its history.",
        manufacture="New bodies use new specimen IDs. List every preparation a manufacture request uses, including a "
                    "reused one with identical conditions.",
        measurement_selection="Specify a nonempty measurements list for each specimen or request.",
        requests="request_simulation(simulation='magnetic') executes requests on this lot and returns at once. "
                 "submit_batch records physical proposals only. record_decision(stop) ends the investigation; wait "
                 "also ends it here.",
        simulation="simulation='magnetic' selects this investigation's lot. Omitting simulation reaches a separate legacy "
                   "simulator of an older historical lot, which is not part of this investigation.")
    task["semantics"].pop("inventory_label", None)
    task["hybrid_simulation"]["scope_note"] = ("Legacy simulator (simulation omitted): a different, older lot; not this "
                                               "investigation's lot.")
    packet["reviewed_public_sources"] = [s for s in packet["reviewed_public_sources"]
                                         if not s["content"].startswith("# Release R0 (t0): context")]
    save(ep / "sources.json", packet)
    sim_seed = secrets.randbits(62)   # reading noise; independent of the world seed
    template = json.loads((LAB / "template/investigation/simulation-magnetic/owner_simulation_runtime.json").read_text())
    cfg = copy.deepcopy(template)
    cfg.update(python=str(OWNER_ENV / "bin/python"), runtime=str(OWNER_ENV / "native-runtime"),
               seed=sim_seed, world_config=world, public_capabilities=capabilities(world))
    save(ep / "simulation-magnetic/owner_simulation_runtime.json", cfg)
    save(ep / "simulation-magnetic/reference_world.json", world)
    save(ep / "INITIAL_REQUESTS.json", [reference_request()])
    for d in ("api", "pi_sessions"):
        (ep / "pi" / d).mkdir()
    save(ep / "PREPARATION.json", {
        "version": "sdl2-v1.3-dev.1", "origin": origin, "sim_seed": sim_seed, "data_cutoff": "P7", "new_session": True, "pi": "1.4.0",
        "world_sha256": sha(ep / "simulation-magnetic/reference_world.json"),
        "contract_sha256": sha(ws / "START_HERE.md"), "contract_source": contract_path.name, "laboratory_sha256": sha(ws / "LABORATORY.md"),
        "base_ledger_sha256": sha(BASE / "campaign/campaign.sqlite3"), "training": False})
    print(json.dumps({"episode": str(ep), "workspace": str(ws), "origin": origin}))
    return ep


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("name")
    a.add_argument("--panel")
    a.add_argument("--state")
    a.add_argument("--seed", type=int)
    a.add_argument("--root")
    a.add_argument("--contract")
    x = a.parse_args()
    prepare(x.name, x.panel, x.state, x.seed, x.root, x.contract)
