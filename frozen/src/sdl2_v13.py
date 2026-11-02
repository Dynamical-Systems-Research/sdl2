"""SDL-2 v1.3 owner world: the v1.2 world family with the observation and world-identity leaks closed.

Owner only. v1.2 (frozen) stays the physics: process, density, anneal, magnetic paths, the 30 kOe bound, truth and
grading are its functions, called unchanged on the v1.2-shaped configuration this module carries. v1.3 changes what an
agent can read and where worlds come from (LEAK_FIXES.md has the evidence for each):

  a. Density readings report the method's nominal reading sd, the same for every lot, not the world's own sd. v1.2
     returned the anneal draw's c7_sd_* x 7.62 at full precision, which pins the world's anneal row (and through the
     seed, every draw) from the free reference body.
  b. Hk/Hci is read, not disclosed: the knee field takes the 2 % per-acquisition field error Hci already has, plus a 2 %
     per-piece demagnetization-correction error. (BH)max is recomputed from the read curve. Hci and Br readings are
     bit-identical to v1.2.
  c. The SEM width sigma_ln behind d90 and the coarse fraction is read per request with a 10 % log error, not copied
     from the world's c6_sig (v1.2 returned it exactly through d90 / mean).
  d. The free SPS log's geometry-derived densities carry a 2 % per-log geometric error (v1.2 returned the body's
     exact as-made density at no cost).
  e. Worlds are drawn by draw() from rows the agent never receives: density and magnetic rows from the owner's
     train/eval split of the C7.3 posterior (removed from every shipped copy), anneal parameters from secret-seeded
     refits, and the world seed from secrets.

Hidden values, truth, route outcomes and the grader's estimand are unchanged for a given world; only readings differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import statistics
from pathlib import Path

import numpy as np

from ndfeb_sps.models import c7_campaign as c7
from ndfeb_sps.models import sdl2_v11 as v11
from ndfeb_sps.models import sdl2_v12 as v12

FAMILY = "ndfeb_sdl2_v13"
VERSION = "sdl2-v1.3-dev.1"
STUDY = Path(__file__).resolve().parents[1]
OWNER = STUDY / "owner"

# a. Posterior RMS of the reading sd over the shipped anneal sample (mechanism-weighted): sqrt(E[c7_sd_geo^2]) x 7.62 =
# 0.100 and sqrt(E[c7_sd_arch^2]) x 7.62 = 0.031 g/cm3 (build/NOMINAL_SD.json). Both Archimedes protocols share it.
NOMINAL_DENSITY_SD_G_CM3 = {"geometric": 0.10, "archimedes": 0.031, "archimedes_saturated": 0.031}
HK_ACQ_LOG_SD, HK_PIECE_LOG_SD = 0.02, 0.02   # b. knee-field reading error; per-piece demagnetization correction
SEM_SIGMA_LOG_SD = 0.10                       # c. width read from one field set
REFERENCE_BODY = "REF-B1"                     # the free reference body (env/prepare_v13.py REFERENCE)
SPS_GEOMETRY_LOG_SD = 0.02                    # d. final-geometry calibration of one log
DENSITY_METHOD_TEXT = {
    "archimedes": "archimedes: the piece's density plus a share of its open porosity counted as solid (unsealed "
                  "immersion; fixed per lot)",
    "archimedes_saturated": "archimedes_saturated: dry, saturated and suspended weights; the bulk density of the piece "
                            "with the Archimedes reading scatter, no open-porosity error",
    "geometric": "geometric: mass over cylinder volume, with a relative bias fixed per lot (rough faces, chipped edges "
                 "and graphite foil overstate the volume)"}


def _key(*parts):
    return hashlib.sha256(c7._canonical(list(parts))).hexdigest()


def _v12(world_config):
    if world_config.get("family") != FAMILY:
        raise ValueError("wrong SDL-2 world family")
    return world_config["v12"]


# ------------------------------------------------------------------ readings

def _density(row):
    method = row.get("density_method", "archimedes")
    row["uncertainty"] = {"reading_sd_g_cm3": NOMINAL_DENSITY_SD_G_CM3[method]}
    row["uncertainty_basis"] = ("nominal reading sd of this method, the same for every lot; this lot's actual "
                                "reading scatter is not stated and may differ. The body's deviation and the piece's "
                                "offset from its body are part of the specimen, not of this reading")
    if method in DENSITY_METHOD_TEXT:
        row["method_model"] = DENSITY_METHOD_TEXT[method]


def _loop(world, row, hidden, key, sid, piece):
    """Read the knee: hk_field = hk x Hci x exp(0.02 z_acquisition + 0.02 u_piece); Hk/Hci and (BH)max from the read
    curve. The same acquisition gives the same z on every row it fills."""
    readings = [r for r in (row.get("loop_reading"), row.get("value")) if isinstance(r, dict) and "hk_over_hci" in r]
    if not readings:
        return
    z = float(world.stream7("sdl2-v13", "hk-reading", key).standard_normal())
    u = float(world.stream7("sdl2-v13", "hk-piece", sid, piece).standard_normal())
    hk_field = hidden["hk_over_hci"] * hidden["hci_kOe"] * math.exp(HK_ACQ_LOG_SD * z + HK_PIECE_LOG_SD * u)
    for r in readings:
        hk = min(max(hk_field / r["hci_kOe"], 0.05), 0.95)
        r["hk_over_hci"] = hk
        r["bhmax_MGOe"] = float(world.link._bhmax(np.array([r["br_kGs"]]), np.array([r["hci_kOe"]]),
                                                  np.array([hk]))[0])
    if row["quantity"] == "bhmax_MGOe":
        row["value"] = readings[0]["bhmax_MGOe"]
    unc = row.get("uncertainty") if isinstance(row.get("uncertainty"), dict) else {}
    row["uncertainty"] = dict(unc, log_sd_hk_field=HK_ACQ_LOG_SD,
                              hk_piece_error="a demagnetization-correction error fixed per piece (not stated)")


def _sem_width(world, rows, sid, request_key):
    """d90 and the coarse fraction of one request: the same field set, now with a read width sigma_r."""
    ms = world.ms
    sigma = float(world.c6["c6_sig"][0])
    sigma_r = sigma * math.exp(SEM_SIGMA_LOG_SD * float(world.stream7("sdl2-v13", "sem-width", request_key,
                                                                       sid).standard_normal()))
    unit = float(ms.d90(np.array([1.0]), np.array([sigma]))[0])   # d90 is linear in the median
    median = None
    for row in rows:
        if row["quantity"] == "grain_d90_nm" and row.get("value") is not None:
            median = row["value"] / 1000.0 / unit
    for row in rows:
        if row["quantity"] == "coarse_number_fraction" and median is None and row.get("value") is not None:
            p = float(row["value"])
            if 0.0 < p < 1.0:
                t = float(row["coarse_threshold_nm"]) / 1000.0
                median = math.exp(math.log(t) + sigma * statistics.NormalDist().inv_cdf(p))
    if median is None:
        return
    mean = median * math.exp(sigma ** 2 / 2)
    median_r = mean * math.exp(-sigma_r ** 2 / 2)
    for row in rows:
        if row["quantity"] == "grain_d90_nm" and row.get("value") is not None:
            row["value"] = float(ms.d90(np.array([median_r]), np.array([sigma_r]))[0]) * 1000.0
        elif row["quantity"] == "coarse_number_fraction" and row.get("value") is not None:
            t = float(row["coarse_threshold_nm"]) / 1000.0
            row["value"] = float(ms.coarse_fraction(np.array([median_r]), np.array([sigma_r]), t)[0])
        else:
            continue
        row["uncertainty"] = {"width_log_sd": SEM_SIGMA_LOG_SD}
        row["uncertainty_basis"] = ("the same field-sampled reading as grain_size_nm (declared field log-sd); the "
                                    "distribution width is read from this field set with its own scatter")


def _sps(world, row, sid):
    """Scale every geometry-derived density of one SPS log by one calibration factor g (displacement, temperature and
    timing are recorded as before)."""
    v = row.get("value")
    if not isinstance(v, dict) or "end_relative_density" not in v:
        return
    g = math.exp(SPS_GEOMETRY_LOG_SD * float(world.stream7("sdl2-v13", "sps-geometry", sid).standard_normal()))
    v["relative_density"] = [None if x is None else round(x * g, 5) for x in v["relative_density"]]
    v["end_relative_density"] = v["end_relative_density"] * g
    v["end_density_g_cm3"] = v["end_density_g_cm3"] * g
    v["reduction"]["end_relative_density"] = v["reduction"]["end_relative_density"] * g
    for k in ("relative_density_at_450_c", "pre_load_densification_points_TD"):
        if v.get("events", {}).get(k) is not None:
            v["events"][k] = v["events"][k] * g
    v.setdefault("limits", []).append("Geometry-derived densities carry a per-log calibration error of about 2 % "
                                      "(final dimensions and mass); displacement and temperature are as recorded")


def simulate_sdl2_v13(request, *, world_config, seed):
    """v1.2 execution, then the v1.3 readings (a-d). Request validation, identities and accounting are v1.2's, except
    that the reference body is read once (D5; v1.2 let agents anneal and re-measure it free, as a 13th body)."""
    if request.get("operation") in ("anneal", "measure") and any(
            s.get("specimen_id") == REFERENCE_BODY for s in request.get("specimens", [])):
        raise ValueError(f"{REFERENCE_BODY} is this lot's reference body, read once at initialization; it is not "
                         "annealed or measured again")
    cfg = _v12(world_config)
    delivery = v12.simulate_sdl2_v12(request, world_config=cfg, seed=seed)
    world = c7.World(cfg["base_world"])
    states = {sid: (context, piece, steps, state) for sid, context, piece, steps, state in v11.request_states(request, world)}
    specs = {s["specimen_id"]: s for s in request["specimens"]}
    request_key = _key("request", request)
    sem = {}
    for row in delivery["observations"]:
        sid, quantity = row["specimen_id"], row["quantity"]
        if sid not in states:
            continue
        context, piece, steps, state = states[sid]
        if quantity == "density_g_cm3":
            _density(row)
        elif quantity in c7.MAGNETIC:
            hidden, _ = v12.bounded_state(world, state, sid, piece, steps, path=cfg["magnetic_path"])
            key = _key("loop", request["lot_id"], sid, specs[sid]["preparation_id"], specs[sid]["cycle_id"], piece,
                       state["tool_stage"], context.get("field_set_id"))
            _loop(world, row, hidden, key, sid, piece)
        elif quantity in ("grain_d90_nm", "coarse_number_fraction"):
            sem.setdefault(sid, []).append(row)
        elif quantity == "sps_log":
            _sps(world, row, sid)
    for sid, rows in sem.items():
        _sem_width(world, rows, sid, request_key)
    delivery["provenance"].update(world_version=VERSION, world_family=FAMILY,
                                  request_world_seed_sha256=hashlib.sha256(c7._canonical(
                                      {"request": request, "world": world_config, "seed": seed})).hexdigest())
    return delivery


def truth_for(request, world_config):
    return v12.truth_for(request, _v12(world_config))


def world_truth(world_config, **kw):
    return v12.world_truth(_v12(world_config), **kw)


def route_outcome(world_config, route):
    return v12.route_outcome(_v12(world_config), route)


# ------------------------------------------------------------------ worlds from rows no agent receives

def world_package(split):
    """The owner's world package for a split: the full C7.3 posterior_tool.npz (all rows; worlds use only the split's)
    and the split's secret-seeded anneal store (build/world_packages.py)."""
    return OWNER / f"package-world-{split}"


LOT = "SDL2-LOT"   # the agent-visible lot id, as v1.2 (worlds/worlds.py)


def draw(split, world_id, measurement, *, lot_id=LOT, drawn_at=None, path=None, world_seed=None):
    """One v1.3 world. The seed is secret (secrets.randbits(63)) unless given. Mechanisms by the package's weights, rows
    uniformly within the split's reserved rows (identical rows count with their multiplicity, as the posterior's own
    resampling does), the load-timing label from the row's own probabilities, the anneal mechanism by the fresh store's
    weights and one of its rows, the magnetic path at equal odds."""
    if split not in ("train", "eval"):
        raise ValueError("split is train or eval")
    rows = json.loads((OWNER / "WORLD_SPLIT.json").read_text())["split"]
    pkg_dir = world_package(split)
    package = c7.package_record(pkg_dir)
    _, pm = c7.load_package(package)
    world_seed = secrets.randbits(63) if world_seed is None else int(world_seed)
    rng = np.random.default_rng(world_seed)

    def pick(kind):
        mechs = [m for m in pm._mechs(kind) if pm.w[kind][m] > 0]
        w = np.array([pm.w[kind][m] for m in mechs], float)
        m = mechs[int(rng.choice(len(mechs), p=w / w.sum()))]
        return m, int(rng.choice(rows[m][split]))
    dm, di = pick("density")
    mm, mi = pick("magnetic")
    lt = str(pm._assign_indicator("load_timing", np.array([mm]), np.array([mi]), rng)[0])
    amechs = [m for m in pm.an.ANNEAL_MECHS if m in pm.an_post]
    aw = np.array([pm.an_meta["mechanism_weights"][m] for m in amechs], float)
    am = amechs[int(rng.choice(len(amechs), p=aw / aw.sum()))]
    ai = int(rng.integers(0, len(next(iter(pm.an_post[am].values())))))
    if path is None:
        path = ("link", "c5")[int(rng.integers(0, 2))]
    declared = {"magnetic_index": {k: float(v[mi]) for k, v in sorted(pm.post[mm].items()) if k.startswith(("c5_", "fd_"))},
                "density_index": {k: float(v[di]) for k, v in sorted(pm.post[dm].items()) if k.startswith("c5_")},
                "c6_index": {k: float(pm.c6post[mm][k][mi]) for k in pm.ms.C6_PARAMS},
                "anneal_index": {k: float(v[ai]) for k, v in sorted(pm.an_post[am].items())}}
    block = {k: float(measurement[k]) for k in c7.MEASUREMENT_KEYS[:1]}
    block["link_model_draws"] = int(measurement["link_model_draws"])
    base = {"family": c7.FAMILY, "world_id": world_id, "package": package, "world_seed": world_seed,
            "draw": {"density": {"mechanism": dm, "index": di}, "magnetic": {"mechanism": mm, "index": mi},
                     "load_timing": lt, "anneal": {"mechanism": am, "index": ai}},
            "declared_layer": declared, "lots": [str(lot_id)], "measurement": block,
            "provenance": {"drawn_at": drawn_at, "procedure": "sdl2_v13.draw: secret seed; rows reserved for worlds "
                           f"({split} split); anneal from the {split} secret-seeded refit"}}
    c7.World(base)   # validates the record against the package
    cfg12 = {"family": v12.FAMILY, "version": v12.VERSION, "base_world": base, "magnetic_path": path,
             "magnetic_log_sd": .02, "requirement": copy.deepcopy(v12.REQUIREMENT)}
    return {"family": FAMILY, "version": VERSION, "split": split, "v12": cfg12}
