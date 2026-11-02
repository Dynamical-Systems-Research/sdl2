"""C6 microstructure layer: declared link and measurement functions (pure numpy; no randomness except through the
generators callers pass). Shared by the physics tool C6 (ndfeb_physics_tool.py), the owner calibration
(fit/c6_calibrate.py) and the campaign twin. Nothing here changes a C5 quantity: every function reads C5 latents (the
model grain of a stage, the continuity index W, the body's relative density) and the C6 per-draw parameters c6_*.

Units: um, relative density (fraction of 7.62 g/cm3), fractions in [0, 1], seconds and deg C for the SPS trace.

A1  grain and feature size (per stage: as sintered, after 900 C/2 h, after 900 C/2 h + 520 C/2 h; the C5 grain law gives
    the 520 C step no growth, so the model grain after 900 equals the grain after 900 + 520)
    grain_sem_dist   log-normal grain distribution an SEM grain-size protocol would report: mean D_stage x c_m6 (D_stage
                     the C5 model grain, c_m6 the method factor), sigma_ln = c6_sig; median = mean x exp(-sigma^2 / 2);
                     d90 = median x exp(1.2816 sigma); coarse number fraction above a threshold t:
                     Phi((ln median - ln t) / sigma); a measured median carries field sampling (log-sd c6_sfield)
    fracture_intercept_um  mean linear intercept on a fracture SE image: D_stage x (2/3) x phi(rho) x c_m6 x exp(e),
                     2/3 the sphere stereology (the metrology's own d3 ~ 1.5 x intercept), phi(rho) the feature factor
                     (c6_phip below ~95 %TD, where fracture SE shows particles and ligaments, 1 for dense bodies; smooth
                     in rho over a 0.005 width), e field sampling ~ N(0, c6_sfield^2), independent per stage
    relief           lightly polished porous surfaces (relief class): D_stage x (2/3) x c6_phir x exp(e), log-sd 0.10;
                     relief-class observations only (they constrain c6_phir and nothing else)
A2  porosity_area_fraction_sem   BSE dark area fraction of a planar section: (1 - rho) x exp(c6_pbias + e),
                     e ~ N(0, 0.5^2) per reading; sections below RHO_PLANAR are not planar (no value)
A3  Nd-rich phase on a planar polished BSE section, latent per draw and stage (dW = W - 0.5, W the C5 continuity
    index as sintered; lam = 0 as sintered, c6_lam900 after 900, 1 after 900 + 520):
      nd_rich_area_fraction  exp(ln c6_vnd + ln(oxide factor) + c6_bf dW + lam c6_dfnd)  (Otsu basis, 26.98 nm/px)
      pocket_ecd_um          exp(ln c6_ecd0 + c6_becd dW + lam c6_decd)        (area-weighted ECD, +27.5 offset basis)
      pocket_share           sigmoid(c6_ps0 + c6_bps dW + lam c6_dps)           (thickness split, t = 0.20 um)
      elongated_share        sigmoid(c6_el0 + c6_bel dW + lam c6_del)           (aspect ratio >= 3)
      faint_line_coverage    sigmoid(c6_fl0 + c6_bfl dW + lam c6_dfl)           (T10 lines, 2.5 um intercept basis)
    the oxide factor is the C5 oxide term exp(-c5_kox x the C4 oxidation penalty), applied only for a declared
    feedstock (1 at 4 um, as in C5). Readings carry declared body-and-field scatter (A3_SCATTER; extra for a lossy
    20 kV JPEG). The anneal transition is paired: the same draw's as-sintered latent, shifted.
A4  sps_trace: descriptive SPS display trace (temperature, ram displacement, derived relative density) whose end state is
    the draw's C5 final density (README: "descriptive; densification kinetics not identified").
"""
from __future__ import annotations

import math

import numpy as np

C6_STAGES = ("as_sintered", "after_900", "after_900_520")
STAGE_ALIASES6 = {"900": "after_900", "900+520": "after_900_520"}
C6_PARAMS = ("c6_sig", "c6_cm6", "c6_phip", "c6_sfield", "c6_phir", "c6_pbias", "c6_vnd", "c6_bf", "c6_dfnd",
             "c6_ecd0", "c6_becd", "c6_decd", "c6_ps0", "c6_bps", "c6_dps", "c6_el0", "c6_bel", "c6_del", "c6_fl0",
             "c6_bfl", "c6_dfl", "c6_lam900")
A3_KEYS = ("nd_rich_area_fraction", "pocket_ecd_um", "pocket_share", "elongated_share", "faint_line_coverage")
A3_PARAMS = {"nd_rich_area_fraction": ("c6_vnd", "c6_bf", "c6_dfnd"), "pocket_ecd_um": ("c6_ecd0", "c6_becd", "c6_decd"),
             "pocket_share": ("c6_ps0", "c6_bps", "c6_dps"), "elongated_share": ("c6_el0", "c6_bel", "c6_del"),
             "faint_line_coverage": ("c6_fl0", "c6_bfl", "c6_dfl")}
KAPPA_ST = 2.0 / 3.0            # mean linear intercept / mean 3D size (spheres); the metrology's d3 ~ 1.5 x intercept
RHO_DENSE_FRACTURE = 0.95       # below ~95 %TD a fracture SE image shows particles and ligaments, not grains (C6_SPEC A1)
FRACTURE_TRANSITION_WIDTH = 0.005
RHO_PLANAR = 0.80               # declared: below 80 %TD a polished section is not planar (no BSE area fraction)
RHO_PLANAR_OBSERVED = 0.92      # the least dense planar section on record is at 92 %TD and the densest non-planar one at
                                # 76 %TD: sections between 0.80 and 0.92 are planar by declaration (no record)
Z90 = 1.2815515655446004
RELIEF_LOG_SD = 0.10            # declared: scatter of one relief-class reading (log-sd)
POROSITY_LOG_SD = 0.5           # C6_SPEC A2: the BSE dark fraction brackets rho-porosity only within a factor of 3-4
POROSITY_FLOOR = 0.002          # 1 - rho is floored here (C5 relative density can reach 1.02)
FL_FLOOR = 0.003                # detection floor of the faint-line coverage (false-positive level at T10)
ND_RICH_MAX = 0.5
# declared body-and-field scatter of one planar-section reading (one body, one field set): (scale, sd). Basis: the
# body-to-body contrasts of the historical and campaign images at near-identical routes, which W alone does not explain
A3_SCATTER = {"nd_rich_area_fraction": ("log", 0.25), "pocket_ecd_um": ("log", 0.40), "pocket_share": ("logit", 0.5),
              "elongated_share": ("logit", 0.4), "faint_line_coverage": ("logfloor", 1.0)}
# extra scatter of a lossy 20 kV JPEG (D's only image: larger interaction volume, upsampled frame), added in quadrature
IMAGE_EXTRA = {"native": {"log": 0.0, "logit": 0.0, "logfloor": 0.0},
               "jpeg_20kV": {"log": 0.35, "logit": 0.5, "logfloor": 0.5}}


def stage6(s):
    s = STAGE_ALIASES6.get(s, s)
    if s not in C6_STAGES:
        raise ValueError(f"stage must be one of {C6_STAGES} ('900' and '900+520' are accepted)")
    return s


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def std_normal_cdf(z):
    z = np.asarray(z, float)
    return 0.5 * np.frompyfunc(math.erfc, 1, 1)(-z / math.sqrt(2.0)).astype(float)


def stage_lambda(p, stage):
    """Share of the 900 + 520 C visible transition present at `stage` (0 as sintered, c6_lam900 after 900, 1 after
    900 + 520), per draw."""
    stage = stage6(stage)
    lam = np.asarray(p["c6_lam900"], float)
    if stage == "as_sintered":
        return np.zeros_like(lam)
    if stage == "after_900":
        return lam
    return np.ones_like(lam)


# ------------------------------------------------------------------------------------------------------------- A1
def feature_factor(p, rho):
    """phi(rho): c6_phip on porous bodies (fracture SE shows particles and ligaments), 1 on dense ones, smooth in ln."""
    s = sigmoid((np.asarray(rho, float) - RHO_DENSE_FRACTURE) / FRACTURE_TRANSITION_WIDTH)
    return np.exp(np.log(np.asarray(p["c6_phip"], float)) * (1.0 - s))


def fracture_latent(p, D, rho):
    """Mean fracture-SE intercept (um) before field sampling: D x 2/3 x phi(rho) x c_m6."""
    return np.asarray(D, float) * KAPPA_ST * feature_factor(p, rho) * np.asarray(p["c6_cm6"], float)


def relief_latent(p, D):
    return np.asarray(D, float) * KAPPA_ST * np.asarray(p["c6_phir"], float)


def grain_median(p, D):
    """Median of the SEM-protocol grain distribution: mean D x c_m6, median = mean x exp(-sigma^2 / 2)."""
    s = np.asarray(p["c6_sig"], float)
    return np.asarray(D, float) * np.asarray(p["c6_cm6"], float) * np.exp(-0.5 * s * s)


def d90(median, sigma):
    return np.asarray(median, float) * np.exp(Z90 * np.asarray(sigma, float))


def coarse_fraction(median, sigma, threshold_um):
    """Number fraction of grains above threshold_um under the log-normal (median, sigma)."""
    t = float(threshold_um)
    return std_normal_cdf((np.log(np.asarray(median, float)) - math.log(t)) / np.asarray(sigma, float))


# ------------------------------------------------------------------------------------------------------------- A2
def planar(rho):
    return np.asarray(rho, float) >= RHO_PLANAR


def porosity_latent(rho):
    return np.clip(1.0 - np.asarray(rho, float), POROSITY_FLOOR, 1.0)


# ------------------------------------------------------------------------------------------------------------- A3
def a3_latent_one(p, W, lam, key, oxide_factor=None):
    """Latent A3 quantity `key` for draws with continuity W and stage share lam (0 as sintered, 1 after 900 + 520);
    p needs only that key's three parameters (A3_PARAMS[key]). Arrays broadcast."""
    a, b, d = (np.asarray(p[k], float) for k in A3_PARAMS[key])
    dW = np.asarray(W, float) - 0.5
    lam = np.asarray(lam, float)
    if key == "nd_rich_area_fraction":
        lnf = np.log(a) + b * dW + lam * d
        if oxide_factor is not None:
            lnf = lnf + np.log(np.maximum(np.asarray(oxide_factor, float), 1e-12))
        return np.minimum(np.exp(lnf), ND_RICH_MAX)
    if key == "pocket_ecd_um":
        return np.exp(np.log(a) + b * dW + lam * d)
    return sigmoid(a + b * dW + lam * d)


def a3_latent(p, W, stage, oxide_factor=None):
    """Latent A3 quantities per draw at `stage` (dict of arrays aligned with W)."""
    lam = stage_lambda(p, stage)
    return {k: a3_latent_one(p, W, lam, k, oxide_factor) for k in A3_KEYS}


def _scale(key, image="native"):
    kind, s = A3_SCATTER[key]
    extra = IMAGE_EXTRA[image][kind]
    return kind, math.hypot(s, extra)


def a3_measure(key, x, z, image="native"):
    """One simulated reading of A3 quantity `key` from latent x and standard normals z (same shape)."""
    kind, s = _scale(key, image)
    x = np.asarray(x, float)
    if kind == "log":
        y = x * np.exp(s * z)
        return np.minimum(y, 1.0) if key == "nd_rich_area_fraction" else y
    if kind == "logit":
        return sigmoid(logit(x) + s * z)
    return np.clip((x + FL_FLOOR) * np.exp(s * z) - FL_FLOOR, 0.0, 1.0)


def a3_loglik(key, y, x, image="native"):
    """Log-likelihood of an observed reading y (scalar) under latents x (array)."""
    kind, s = _scale(key, image)
    x = np.asarray(x, float)
    if kind == "log":
        r = (math.log(y) - np.log(x)) / s
    elif kind == "logit":
        r = (float(logit(y)) - logit(x)) / s
    else:
        r = (math.log(max(y, 0.0) + FL_FLOOR) - np.log(x + FL_FLOOR)) / s
    return -0.5 * r * r - math.log(s) - 0.5 * math.log(2 * math.pi)


def normal_loglik(y_log, mu_log, s):
    s = np.asarray(s, float)
    r = (y_log - np.asarray(mu_log, float)) / s
    return -0.5 * r * r - np.log(s) - 0.5 * math.log(2 * math.pi)


# ------------------------------------------------------------------------------------------------------------- A4
# Declared descriptive priors of the SPS display trace (owner fits of RPI's raw logs #1, #2, #4, #5; #3's temperature
# column duplicates #2's and is excluded; microstructure/sintering/PER_BODY_TABLE.csv and data/RECONSTRUCTION.json).
SPS_PRIORS = {
    "rho_450": ("uniform", 0.37, 0.43, "relative density at 450 C under the pre-load (logs: 37.4-42.7 %TD)"),
    "T_onset_C": ("uniform", 505.0, 540.0, "shrinkage onset under the pre-load (logs: 505-540 C; #1 575 C)"),
    "step_share": ("uniform", 0.56, 0.68, "share of the densification from 450 C to the end carried by the load step "
                   "(logs #1, #2, #4, #5: 0.56-0.68)"),
    "heat_share": ("uniform", 0.60, 0.81, "share of the pre-load densification made during heating rather than the "
                   "dwell (logs #1, #2, #4, #5: 0.60-0.81)"),
    "heat_shape": ("uniform", 1.5, 3.0, "exponent of the heating densification in (T - onset) / (T_peak - onset) "
                   "(the ramp rate accelerates again at 760-780 C)"),
    "dwell_tau_min": ("uniform", 1.0, 5.0, "time constant of the dwell densification under the pre-load (min)"),
    "step_duration_s": ("uniform", 18.0, 35.0, "duration of the load step at load_ramp_min 0.3 (logs: 18-35 s); "
                        "scaled by load_ramp_min / 0.3 for other ramps (declared)"),
    "step_lead_s": ("uniform", 3.0, 10.0, "the load step starts this long before the logged temperature falls "
                    "(logs: 3-10 s)"),
    "compliance_mm": ("uniform", 0.0, 0.30, "elastic tooling compliance taken up with the load (bracketed 0-0.3 mm)"),
    "alpha_heat_mm_per_C": ("uniform", 0.00049, 0.00071, "tooling expansion during heating (logs: 0.49-0.71 um/C)"),
    "alpha_cool_mm_per_C": ("uniform", 0.00087, 0.00143, "tooling contraction after power-off (logs: 0.87-1.43 um/C)"),
}
SPS_DIE_DIAMETER_MM = 10.0
SPS_CHARGE_MASS_G = 0.85        # declared nominal charge (RPI's #1-#5: 0.77-0.92 g)
SPS_RHO_TD = 7.62
SPS_T_START_C = 25.0
SPS_T_LOG_END_C = 300.0         # the display trace stops at 300 C or 20 min after power-off
SPS_COOL_MAX_S = 1200.0
SPS_QUANTIZATION = {"t_s": 1.0, "T_C": 0.1, "displacement_mm": 0.01}
SPS_STATUS = "descriptive (from RPI's raw logs #1, #2, #4, #5); final density from C5; densification kinetics not identified"
SPS_STATUS_DECLARED_TIMING = ("declared: load timing other than at power-off; the load step is placed at the requested "
                              "time (no log records it); final density from C5")


def sps_draw(n, rng):
    """n draws of the declared trace parameters (dict of (n,) arrays)."""
    out = {}
    for k, spec in SPS_PRIORS.items():
        out[k] = rng.uniform(spec[1], spec[2], n)
    return out


def _height_mm(rho, mass_g):
    area_cm2 = math.pi * (SPS_DIE_DIAMETER_MM / 20.0) ** 2
    return 10.0 * mass_g / (np.maximum(rho, 1e-3) * SPS_RHO_TD * area_cm2)


def sps_trace(route, rho_end, prm, dt_s=1.0, mass_g=SPS_CHARGE_MASS_G, quantize=True):
    """Descriptive SPS display trace for draws with final relative density rho_end ((n,) array) and trace parameters
    prm (dict of (n,) arrays from sps_draw). route: a resolved tool route (T_peak_C, heating_rate_C_per_min, dwell_min,
    cooling_rate_C_per_min, pressure_mode, load_ramp_min). Returns t_s (T,), T_C (T,), displacement_mm (n, T),
    rel_density (n, T) and per-draw events (evaluated at their exact times, not on the grid). The relative density
    ends exactly at rho_end."""
    rho_end = np.atleast_1d(np.asarray(rho_end, float))
    n = len(rho_end)
    Tp = float(route["T_peak_C"])
    rate = float(route["heating_rate_C_per_min"])
    dwell_s = max(float(route["dwell_min"]) * 60.0, 0.0)
    cool = float(route["cooling_rate_C_per_min"])
    mode = route["pressure_mode"]
    t_heat = max((Tp - SPS_T_START_C) / rate * 60.0, 0.0)
    t_off = t_heat + dwell_s
    k = cool / max(Tp - SPS_T_START_C, 1e-9) / 60.0  # Newtonian cooling, initial slope = the route's cooling rate
    if Tp > SPS_T_LOG_END_C:
        t_cool = min(math.log((Tp - SPS_T_START_C) / (SPS_T_LOG_END_C - SPS_T_START_C)) / k, SPS_COOL_MAX_S)
    else:
        t_cool = 60.0
    c = lambda key: np.asarray(prm[key], float)[:, None]
    r450 = np.minimum(c("rho_450"), rho_end[:, None])
    total = rho_end[:, None] - r450
    ton = c("T_onset_C")
    share = np.where(Tp > ton, c("step_share"), 1.0)  # pre-load densification needs the onset to be reached
    d_step = share * total
    d_pre = total - d_step
    d_heat = c("heat_share") * d_pre
    d_dwell = d_pre - d_heat
    tau = c("dwell_tau_min") * 60.0
    dur = c("step_duration_s") * (float(route.get("load_ramp_min", 0.3)) / 0.3)
    if mode == "cooling":
        t0 = t_off - c("step_lead_s")
    else:  # declared: load in heating reaches its plateau at T_peak (end of the ramp); load from the dwell start
        t0 = np.full((n, 1), t_heat)
    done = np.maximum(t0 + dur, t_off)
    # the display runs to 300 C (or 20 min after power-off), and at least until every draw's load step has finished
    t = np.arange(0.0, max(t_off + t_cool, float(done.max()) + 10.0) + 0.5 * dt_s, dt_s)
    h0 = _height_mm(r450, mass_g)

    def temp(tt):
        T = np.where(tt <= t_heat, SPS_T_START_C + rate * tt / 60.0, Tp)
        T = np.where(tt > t_off, SPS_T_START_C + (Tp - SPS_T_START_C) * np.exp(-k * (tt - t_off)), T)
        return np.minimum(T, Tp)

    def state(tt):  # tt: (n, K) or (1, K) times; returns T, rel, disp of shape (n, K)
        T = temp(tt)
        x_h = np.clip((T - ton) / np.maximum(Tp - ton, 1e-9), 0.0, 1.0)
        x_h = np.where(tt <= t_heat, x_h, 1.0)
        heat = d_heat * x_h ** c("heat_shape")
        if dwell_s > 0:
            dw = d_dwell * (-np.expm1(-np.clip(tt - t_heat, 0.0, dwell_s) / tau)) / (-np.expm1(-dwell_s / tau))
        else:
            dw = np.where(tt >= t_heat, d_dwell, 0.0)
        gs = 1.0 - (1.0 - np.clip((tt - t0) / dur, 0.0, 1.0)) ** 2  # ~75 % within half the step (logs: 75-80 %)
        rel = np.where(tt >= done, rho_end[:, None], r450 + heat + dw + d_step * gs)
        th = np.where(tt <= t_off, -c("alpha_heat_mm_per_C") * (T - SPS_T_START_C),
                      -c("alpha_heat_mm_per_C") * (Tp - SPS_T_START_C) + c("alpha_cool_mm_per_C") * (Tp - T))
        disp = (h0 - _height_mm(rel, mass_g)) + c("compliance_mm") * gs + th
        return np.broadcast_to(T, rel.shape), rel, disp

    T, rel, disp = state(np.broadcast_to(t[None, :], (n, len(t))))
    rel = np.array(rel)
    rel[:, -1] = rho_end
    _, _, d0 = state(t0)
    _, _, d1 = state(t0 + dur)
    events = {"t_power_off_s": float(t_off), "t_heating_end_s": float(t_heat),
              "step_start_s": t0[:, 0], "step_duration_s": dur[:, 0], "step_dD_mm": (d1 - d0)[:, 0],
              "step_share": share[:, 0], "pre_load_points_TD": 100.0 * d_pre[:, 0],
              "rho_450": r450[:, 0], "onset_C": ton[:, 0]}
    T = temp(t)
    if quantize:
        T = np.round(T / SPS_QUANTIZATION["T_C"]) * SPS_QUANTIZATION["T_C"]
        disp = np.round(disp / SPS_QUANTIZATION["displacement_mm"]) * SPS_QUANTIZATION["displacement_mm"]
    return {"t_s": t, "T_C": T, "displacement_mm": np.asarray(disp), "rel_density": rel, "events": events}
