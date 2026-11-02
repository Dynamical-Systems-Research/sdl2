"""C7 anneal layer: post-sinter densification of a body during a load-free anneal, and the density reading an
instrument returns at a stage (pure numpy; no randomness except through the generators callers pass).

Shared by the physics tool C7 (ndfeb_physics_tool.py), the owner calibration (build/c7_anneal_fit.py) and the
campaign twin (runtime c7_campaign.py). Nothing here changes a C5 or C6 quantity of an as-sintered body.

Units: relative density is a fraction of 7.62 g/cm3 (the tool's basis, RHO_TD); porosity phi = 1 - rho; temperature in
deg C at the API; time in hours inside the rate law.

Evidence (P5, October 20): RPI reported body #8 at 7.07 g/cm3 as sintered and 7.58 g/cm3 after a 900 C heat treatment
(92 % / 99 % TD as reported). The return does not state the density method, its uncertainty, the TD denominator, the
anneal hold, atmosphere or cooling, or whether one piece was measured at both stages. One paired change identifies only
the anneal dose of that unconfirmed program under each mechanism below; it cannot rank the mechanisms. They make
different predictions for bodies below about 90 %TD, which is what a retained lower-density piece would test.

Densification mechanisms (ANNEAL_MECHS; each world carries one; the package weights them):
  LPS       liquid-assisted pressureless densification of all porosity: d phi / dt = -k(T) (phi - phi_floor). The
            Nd-rich phase is liquid above its eutectic (onset c7_Tliq), so the rate switches on there.
  CLOSED    final-stage densification of closed porosity only: d phi / dt = -k(T) [f_c(phi) + r_open (1 - f_c(phi))]
            (phi - phi_floor), f_c(phi) = sigmoid((c7_phic - phi) / c7_wc) the closed share; open-pore bodies coarsen
            and barely shrink without load (r_open << 1).
  CAPACITY  shrinkage limited by the liquid available to fill pores: phi -> phi_floor + max(phi0 - phi_floor -
            c7_cap, 0) + (removable part) exp(-k t); at most c7_cap of porosity can be removed by one anneal.
  NULL      no densification (k = 0): the reported change is a method, piece or reporting artifact.
k(T) = exp(c7_lnk900) exp(-(c7_Q / R)(1/T - 1/1173.15 K)) sigmoid((T - c7_Tliq) / c7_wliq), per hour. A 520 C step lies
below the eutectic onset and adds no densification in every mechanism (declared).

Density reading (DENSITY_METHODS): what a density measurement of one piece returns,
  archimedes   rho_piece + c7_kopen phi_open(rho_piece) + e, e ~ N(0, c7_sd_arch): an unsealed immersion reading
               counts water-filled open porosity as solid (c7_kopen in [0, 1]: 0 sealed or saturated-weight method, 1 no
               correction); phi_open = phi (1 - f_c(phi)) with the CLOSED mechanism's transition
  geometric    rho_piece (1 + c7_bgeo) + e, e ~ N(0, c7_sd_geo): mass over cylinder volume; rough faces, edge chips
               and graphite residue bias the volume
  unspecified  a reading whose method is not stated: an equal mixture of the two (the draw picks one per reading)
  archimedes_saturated  (C7.1) dry, water-saturated and suspended weights give the bulk density of a porous piece
               without the open-porosity error: rho_piece + e, e ~ N(0, c7_sd_arch)
rho_piece = rho_body + eta, eta ~ N(0, c7_sdpiece^2) per piece (radial and axial gradients of an SPS disk); the same
piece measured at two stages shares eta, different pieces do not.
"""
from __future__ import annotations

import math

import numpy as np

ANNEAL_MECHS = ("LPS", "CLOSED", "CAPACITY", "NULL")
# where each mechanism comes from (the investigators' post-P5 forms are special cases; their agreement on #8 is built in,
# all three fit the same single pair)
MECH_SOURCES = {"LPS": "DeepSeek v7 (rho + 0.90 (0.995 - rho)) and Opus v2 M-LPS (10 % of porosity remains)",
                "CLOSED": "Opus v2 M-open (open porosity below ~90 %TD densifies little without load); Qwen's "
                          "pre-registered v1.7 open/closed threshold",
                "CAPACITY": "Qwen v1.6 fixed increment (+0.51 g/cm3, capped) as the limit of a bounded removable "
                            "porosity",
                "NULL": "method, piece and reporting explanations raised by all three investigators and the operator"}
DENSITY_METHODS = ("archimedes", "archimedes_saturated", "geometric", "unspecified")
R_GAS = 8.314462618e-3   # kJ / (mol K)
T_REF_K = 1173.15        # 900 C
N_STEPS_ODE = 64         # least number of sub-steps per step for the CLOSED rate (piecewise-frozen closed share)
CLOSED_MAX_SUBDOSE = None  # C7.1: unused (CLOSED is solved through its dose integral, _closed_solve); kept so the
                           # C7.1 constants list stays stable
ANNEAL_PARAMS = ("c7_lnk900", "c7_Q", "c7_Tliq", "c7_wliq", "c7_phifloor", "c7_phic", "c7_wc", "c7_ropen", "c7_cap",
                 "c7_rhofull", "c7_hold8_h")
MEASURE_PARAMS = ("c7_kopen", "c7_sd_arch", "c7_bgeo", "c7_sd_geo", "c7_sdpiece")
# the #8 program: one load-free step at 900 C whose hold RPI has not stated. A dose-identified prediction for "the same
# program as #8" needs no hold; an explicit program needs the hold #8 actually had (c7_hold8_h, a declared scenario).
PROGRAM_AS_8 = "as_#8_900C"
PLANNED_HOLD_H = 2.0     # the September timeline's 900 C / 2 h program (planned, not confirmed for #8)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def rate(p, T_C):
    """k(T) per hour for draws p (arrays aligned with the draws)."""
    T = np.asarray(T_C, float) + 273.15
    arr = np.exp(-(np.asarray(p["c7_Q"], float) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    liq = sigmoid((np.asarray(T_C, float) - np.asarray(p["c7_Tliq"], float)) / np.asarray(p["c7_wliq"], float))
    return np.exp(np.asarray(p["c7_lnk900"], float)) * arr * liq


def closed_share(p, phi):
    """Closed share f_c(phi) of the porosity (1 for dense bodies, 0 for open compacts)."""
    return sigmoid((np.asarray(p["c7_phic"], float) - np.asarray(phi, float)) / np.asarray(p["c7_wc"], float))


def _step(mech, p, phi, dose, state):
    """Porosity after one isothermal step of dose k t (hours x per hour), per draw. mech: array of labels. state carries
    what a step sequence shares: the porosity floor and, for CAPACITY, the removable porosity still left (set once from
    the as-sintered body and depleted by every step, so splitting a program into segments removes nothing extra)."""
    phi = np.asarray(phi, float)
    floor = state["floor"]
    out = phi.copy()
    lps = mech == "LPS"
    out = np.where(lps, floor + (phi - floor) * np.exp(-dose), out)
    cap = mech == "CAPACITY"
    removed = state["removable"] * -np.expm1(-dose)
    out = np.where(cap, phi - removed, out)
    state["removable"] = np.where(cap, state["removable"] - removed, state["removable"])
    cl = mech == "CLOSED"
    if np.any(cl):
        out = np.where(cl, _closed_solve(p, phi, floor, dose, cl), out)
    return np.where(mech == "NULL", phi, out)


# C7.1: CLOSED solved through its dose integral. dx/dD = -g(x) (x - floor), g = f_c(x) + r_open (1 - f_c(x)), is
# autonomous and separable: with u = ln(x - floor), dD = -du / g. G(u) = int_{U_LO}^{u} du' / g is tabulated per draw on
# a fixed u grid (trapezoid) and the state after a dose D is G^-1(G(u0) - D) by linear interpolation, so the result depends
# only on the total dose: splitting a program into segments gives the same porosity to rounding.
U_LO, U_HI, U_N = math.log(1e-9), 0.0, 2048
_U_GRID = np.linspace(U_LO, U_HI, U_N)
_CHUNK = 16384


def _closed_solve(p, phi, floor, dose, mask):
    phi, floor, dose = (np.broadcast_to(np.asarray(a, float), np.shape(mask)) for a in (phi, floor, dose))
    out = phi.copy()
    idx = np.flatnonzero(mask)
    du = _U_GRID[1] - _U_GRID[0]
    for a in range(0, len(idx), _CHUNK):
        j = idx[a:a + _CHUNK]
        par = {k: np.broadcast_to(np.asarray(p[k], float), np.shape(mask)).ravel()[j][:, None]
               for k in ("c7_phic", "c7_wc", "c7_ropen")}
        fl = floor.ravel()[j]
        x = fl[:, None] + np.exp(_U_GRID)[None, :]
        f = sigmoid((par["c7_phic"] - x) / par["c7_wc"])
        inv_g = 1.0 / (f + par["c7_ropen"] * (1.0 - f))
        G = np.concatenate([np.zeros((len(j), 1)), np.cumsum(0.5 * (inv_g[:, 1:] + inv_g[:, :-1]) * du, 1)], 1)
        e0 = phi.ravel()[j] - fl
        live = e0 > math.exp(U_LO)
        u0 = np.log(np.maximum(e0, math.exp(U_LO)))
        k0 = np.clip(((u0 - U_LO) / du).astype(int), 0, U_N - 2)
        t0 = (u0 - _U_GRID[k0]) / du
        rows = np.arange(len(j))
        g0 = G[rows, k0] + t0 * (G[rows, k0 + 1] - G[rows, k0])
        tgt = g0 - dose.ravel()[j]
        k1 = np.clip((G < tgt[:, None]).sum(1) - 1, 0, U_N - 2)
        span = np.maximum(G[rows, k1 + 1] - G[rows, k1], 1e-300)
        u1 = _U_GRID[k1] + du * np.clip((tgt - G[rows, k1]) / span, 0.0, 1.0)
        x1 = np.where(tgt <= 0.0, fl, fl + np.exp(u1))
        out.ravel()[j] = np.where(live, np.minimum(x1, phi.ravel()[j]), phi.ravel()[j])
    return out


def anneal_porosity(p, mech, phi0, steps):
    """Porosity after a sequence of steps [(T_C, hold_h), ...] for draws p with mechanism labels mech, starting from the
    as-sintered porosity phi0 (the whole executed sequence in one call: the CAPACITY state starts there)."""
    phi = np.asarray(phi0, float)
    mech = np.broadcast_to(np.asarray(mech).astype(str), phi.shape)
    floor = np.minimum(np.asarray(p["c7_phifloor"], float), phi)
    state = {"floor": floor, "removable": np.minimum(phi - floor, np.asarray(p["c7_cap"], float))}
    for T_C, hold_h in steps:
        phi = _step(mech, p, phi, rate(p, T_C) * np.asarray(hold_h, float), state)
    return phi


def porosity(p, rho):
    """Porosity against the alloy's full density c7_rhofull (relative to 7.62; RPI's TD denominator is not stated)."""
    return np.clip(1.0 - np.asarray(rho, float) / np.asarray(p["c7_rhofull"], float), 0.0, 1.0)


def anneal_density(p, mech, rho0, steps):
    """Relative density after the steps [(T_C, hold_h), ...] (never below the as-sintered density)."""
    rho0 = np.asarray(rho0, float)
    phi = anneal_porosity(p, mech, porosity(p, rho0), steps)
    return np.maximum(np.asarray(p["c7_rhofull"], float) * (1.0 - phi), rho0)


def anneal_density_as8(p, mech, rho0):
    """Relative density after the same (unconfirmed) program as #8: the dose k(900) x hold that #8 identifies, whatever
    the hold was. Explicit programs (anneal_density) instead need the draw's declared #8 hold c7_hold8_h."""
    return anneal_density(p, mech, rho0, [(900.0, np.asarray(p["c7_hold8_h"], float))])


def piece_density(p, mech, rho_body_as_sintered, initial_piece_offset, steps):
    """Density of a traceable piece after its whole thermal history (C7.2).

    A piece's offset describes its initial local porosity, not an additive instrument bias that survives annealing.
    The same piece keeps the same initial offset; its local density then evolves under the selected mechanism.
    Calibration, experiment selection, updating and twin readings use this identical mapping. It is a declared
    local-response approximation, not a measured account of transport between regions of a compact.
    """
    return anneal_density(p, mech, np.asarray(rho_body_as_sintered) + np.asarray(initial_piece_offset), steps)


def open_porosity(p, rho):
    phi = porosity(p, rho)
    return phi * (1.0 - closed_share(p, phi))


def reading_mean(p, rho_piece, method):
    """Expected reading (relative) of a piece by `method` ('archimedes' or 'geometric'), before reading noise."""
    rho_piece = np.asarray(rho_piece, float)
    if method == "archimedes":
        return np.minimum(rho_piece + np.asarray(p["c7_kopen"], float) * open_porosity(p, rho_piece), 1.02)
    if method == "archimedes_saturated":   # C7.1: dry, saturated and suspended weights (ASTM B962-type): bulk density
        return np.minimum(rho_piece, 1.02)
    if method == "geometric":
        return rho_piece * (1.0 + np.asarray(p["c7_bgeo"], float))
    raise ValueError(f"method must be one of {DENSITY_METHODS[:3]} here")


def reading_sd(p, method):
    return np.asarray(p["c7_sd_geo" if method == "geometric" else "c7_sd_arch"], float)


def read_density(p, rho_piece, method, rng):
    """One density reading per draw; 'unspecified' draws the method per reading (equal odds). Returns (value, method)."""
    rho_piece = np.asarray(rho_piece, float)
    if method not in DENSITY_METHODS:
        raise ValueError(f"density method must be one of {DENSITY_METHODS}")
    if method == "unspecified":
        pick = rng.random(rho_piece.shape) < 0.5
        a = reading_mean(p, rho_piece, "archimedes") + reading_sd(p, "archimedes") * rng.standard_normal(rho_piece.shape)
        g = reading_mean(p, rho_piece, "geometric") + reading_sd(p, "geometric") * rng.standard_normal(rho_piece.shape)
        return np.where(pick, a, g), np.where(pick, "archimedes", "geometric")
    v = reading_mean(p, rho_piece, method) + reading_sd(p, method) * rng.standard_normal(rho_piece.shape)
    return v, np.full(rho_piece.shape, method)


def reading_logpdf(p, y, rho_piece, method):
    """Log density of a reading y (relative) given the piece's true density, per draw; 'unspecified' mixes 50/50."""
    def one(m):
        mu, sd = reading_mean(p, rho_piece, m), reading_sd(p, m)
        return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)
    if method == "unspecified":
        a, g = one("archimedes"), one("geometric")
        m = np.maximum(a, g)
        return m + np.log(0.5 * np.exp(a - m) + 0.5 * np.exp(g - m))
    return one(method)
