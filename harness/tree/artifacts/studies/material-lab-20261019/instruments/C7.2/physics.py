"""Physics core for the SPS NdFeB process -> microstructure -> magnetic chain.

Pure functions. Route schedules are discretised once with numpy (they carry no parameters);
everything that depends on parameters is written in jax.numpy so it can sit inside NumPyro
models, the tool and the world generator alike.

Units: temperature in K inside rate laws (deg C at the API), time in minutes, pressure in MPa,
magnetic quantities in kOe / kGs / MGOe (CGS, the units RPI reports), polarisation J = 4*pi*M.

Backend: every parameter-dependent function takes `xp` (numpy by default, jax.numpy inside NumPyro).
This module never imports jax, so the agent tool runs with numpy/scipy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314462618e-3  # kJ / (mol K)
T_REF_K = 1023.15       # 750 C reference temperature for rate constants
P_REF = 50.0            # MPa reference pressure
N_HEAT, N_DWELL, N_COOL = 160, 40, 80
T_COOL_END_C = 300.0    # densification/desorption below this is negligible on these time scales
COOL_RAMP_MIN = 1.5     # E's slide-12 load ramp (0->50 MPa over 850->700 C); routes carry their own load_ramp_min


@dataclass(frozen=True)
class Route:
    """One SPS route. Fields mirror data/routes.csv; defaults are the current RPI route."""
    T_peak_C: float
    heating_rate_C_per_min: float = 10.0
    dwell_min: float = 5.0
    P_MPa: float = 60.0
    pressure_mode: str = "cooling"          # "heating" | "cooling" | "dwell" (v4: load from the start of the dwell)
    cooling_rate_C_per_min: float = 100.0
    load_ramp_min: float = 0.3               # load-on-cooling ramp 0->P (current RPI traces ~0.3 min; E's slide-12 row 1.5)
    degas: float = 1.0                       # 1 degassed (600 C / 10 min), 0 not, fractional = probability
    anneal: str = "none"                     # "none" | "900" | "900+520"
    alignment_cos: float = 0.5               # <cos theta>; 0.5 = isotropic (declared scenario if != 0.5)
    T_measure_C: float = 20.0                # measurement / operating temperature (declared scenario if != 20)
    prep_id: str = "NEW"
    label: str = ""
    current_path: str = "compact"            # v4: "compact" (loaded compact carries current) | "insulated"


@dataclass
class Schedule:
    T_K: np.ndarray       # (n,)
    P: np.ndarray         # (n,)
    dt: np.ndarray        # (n,) minutes, trapezoid-equivalent weights
    pre: np.ndarray       # (n,) 1 for points before the compaction event, else 0
    hotopen: np.ndarray   # (n,) 1 during heating+dwell (pores open in cooling mode), 0 during cooling
    heating: np.ndarray   # (n,) 1 during the heating ramp
    jump: np.ndarray      # (n,) 1 at the step where the load is first applied (cold-compaction jump)
    meta: dict = field(default_factory=dict)


def _seg(t0, t1, n):
    t = np.linspace(t0, t1, n)
    w = np.full(n, (t1 - t0) / (n - 1)) if n > 1 else np.zeros(1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return t, w


def schedule(r: Route) -> Schedule:
    Tp = float(r.T_peak_C)
    t_heat = max((Tp - 25.0) / r.heating_rate_C_per_min, 1e-3)
    th, wh = _seg(0.0, t_heat, N_HEAT)
    Th = 25.0 + r.heating_rate_C_per_min * th
    td, wd = _seg(0.0, max(r.dwell_min, 1e-3), N_DWELL)
    Td = np.full(N_DWELL, Tp)
    t_cool = max((Tp - T_COOL_END_C) / r.cooling_rate_C_per_min, 1e-3)
    tc, wc = _seg(0.0, t_cool, N_COOL)
    Tc = Tp - r.cooling_rate_C_per_min * tc
    if r.pressure_mode == "heating":
        Ph = r.P_MPa * th / t_heat
        Pd = np.full(N_DWELL, r.P_MPa)
        Pc = np.full(N_COOL, r.P_MPa)
        # load reaches its plateau at T_peak at the end of the ramp: compaction there, dwell under load
        pre = np.concatenate([np.ones(N_HEAT), np.zeros(N_DWELL + N_COOL)])
        jump = np.zeros(N_HEAT + N_DWELL + N_COOL); jump[0] = 1.0
    elif r.pressure_mode == "cooling":
        Ph = np.zeros(N_HEAT)
        Pd = np.zeros(N_DWELL)
        Pc = r.P_MPa * np.clip(tc / max(r.load_ramp_min, 1e-3), 0.0, 1.0)
        pre = np.concatenate([np.ones(N_HEAT + N_DWELL), np.zeros(N_COOL)])
        jump = np.zeros(N_HEAT + N_DWELL + N_COOL); jump[N_HEAT + N_DWELL] = 1.0
    elif r.pressure_mode == "dwell":
        # v4 declared scenario: no load during heating; the load rises 0 -> P over load_ramp_min from the start of the
        # dwell and is held through the dwell and cooling. Never fitted (no record uses it).
        Ph = np.zeros(N_HEAT)
        Pd = r.P_MPa * np.clip(td / max(r.load_ramp_min, 1e-3), 0.0, 1.0)
        Pc = np.full(N_COOL, r.P_MPa)
        pre = np.concatenate([np.ones(N_HEAT), np.zeros(N_DWELL + N_COOL)])
        jump = np.zeros(N_HEAT + N_DWELL + N_COOL); jump[N_HEAT] = 1.0
    else:
        raise ValueError(f"unknown pressure_mode {r.pressure_mode!r}")
    T = np.concatenate([Th, Td, Tc]) + 273.15
    P = np.concatenate([Ph, Pd, Pc])
    dt = np.concatenate([wh, wd, wc])
    hotopen = np.concatenate([np.ones(N_HEAT + N_DWELL), np.zeros(N_COOL)])
    heating = np.concatenate([np.ones(N_HEAT), np.zeros(N_DWELL + N_COOL)])
    return Schedule(T, P, dt, pre, hotopen, heating, jump, meta=dict(t_heat=t_heat, t_cool=t_cool))


def stack(scheds):
    return {k: np.stack([getattr(s, k) for s in scheds]) for k in ("T_K", "P", "dt", "pre", "hotopen", "heating", "jump")}


def sigmoid(x, xp=np):
    return 1.0 / (1.0 + xp.exp(-x))


# ------------------------------------------------------------------------------------------
# Densification (porosity phi, relative density rho = rho_max (1 - phi)).
#   dphi/dt = -[kA(T,P) + kR(T,P) g(phi)] phi (phi/phi0)^(p-1)
#   kA = exp(lnk) exp(-Q/R (1/T - 1/Tref)) ((P/Pref)^n + s) L(T)    densification (diffusion / creep / liquid)
#   kR = exp(lnkR) exp(-QR/R (1/T - 1/Tref)) (P/Pref)                 low-activation particle rearrangement /
#        plastic flow, active only while phi > phiR: g = sigmoid((phi - phiR)/0.02)
# plus an athermal cold-compaction jump phi *= exp(-cR P/Pref) when the load is first applied (start of the ramp in
# heating mode, start of cooling in cooling mode). Integrated exactly per step in u = (phi/phi0)^-(p-1) with k frozen
# over the step (u is monotone, so the scheme is stable for any k dt).
# ------------------------------------------------------------------------------------------
def _scan(step, carry, xs, xp):
    if xp is np:
        for i in range(xs[0].shape[-1]):
            carry = step(carry, tuple(x[..., i] for x in xs))
        return carry
    import jax  # lazy: the agent tool never reaches this branch
    xs_t = tuple(xp.moveaxis(x, -1, 0) for x in xs)
    carry, _ = jax.lax.scan(lambda c, x: (step(c, x), None), carry, xs_t)
    return carry


def _g(x, xp):
    """Lift a per-draw (K,1) or per-route (…,R) parameter to the (…, R, n) time grid."""
    return x[..., None] if xp.ndim(x) else x


def densify(S, mode_cooling, T_peak_C, P_max, prm, xp=np):
    """S: stacked schedule dict (R x n); mode_cooling, T_peak_C, P_max: (R,).
    prm values are scalars (NumPyro) or (K,1) arrays over K posterior draws (tool); prm["off"] is the
    preparation offset on both ln-rates, scalar, (R,) or (K,R). Returns relative density (R,) or (K,R),
    referenced to 7.62 g/cm3."""
    T = S["T_K"]
    off = _g(prm.get("off", 0.0), xp)
    arrA = xp.exp(-(_g(prm["Q"], xp) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    arrR = xp.exp(-(_g(prm["QR"], xp) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    drive = (S["P"] / P_REF) ** _g(prm["n"], xp) + _g(prm["s"], xp)
    liq = 1.0 + _g(prm["lam"], xp) * sigmoid((T - 273.15 - _g(prm["TL"], xp)) / _g(prm["wL"], xp), xp)
    kA = xp.exp(_g(prm["lnk"], xp) + off) * arrA * drive * liq
    kR = xp.exp(_g(prm["lnkR"], xp) + off) * arrR * (S["P"] / P_REF)
    pm1 = prm["p"] - 1.0
    cjump = _g(prm["cR"], xp) * (P_max[..., None] / P_REF) * S["jump"]
    phi0 = prm["phi0"]
    shape = xp.broadcast_shapes(xp.shape(kA)[:-1], xp.shape(kR)[:-1]) if xp is np else kA.shape[:-1]
    u0 = xp.ones(shape)

    def step(u, x):
        kA_i, kR_i, dt_i, j_i = x
        u = u * xp.exp(j_i * pm1)
        phi = phi0 * u ** (-1.0 / pm1)
        gate = sigmoid((phi - prm["phiR"]) / 0.02, xp)
        return u + pm1 * (kA_i + kR_i * gate) * dt_i

    dt = xp.broadcast_to(S["dt"], kA.shape)
    cj = xp.broadcast_to(cjump, kA.shape)
    kR = xp.broadcast_to(kR, kA.shape)
    u = _scan(step, u0, (kA, kR, dt, cj), xp)
    log_phi = xp.log(phi0) - xp.log(u) / pm1
    log_phi = log_phi + prm["d1"] * (T_peak_C - 800.0) / 100.0 + prm["d2"] * mode_cooling
    # porosity can never exceed the loose-powder value (no de-densification), whatever the discrepancy
    log_phi = xp.minimum(log_phi, xp.log(phi0))
    phi = xp.exp(log_phi)
    return xp.clip(prm["rho_max"] * (1.0 - phi), 1e-3, 1.02)


# ------------------------------------------------------------------------------------------
# Hydrogen desorption (H1 and chemistry predictions). Residual hydrogen index after SPS:
#   h_res = h_init * exp(-Theta_des), Theta_des = sum k_des exp(-Qdes/R (1/T - 1/Tref)) * open * dt
# open = 1 while pores are open: always during heating+dwell in cooling mode; in heating mode a
# smooth closure at T_close (pressure-driven closure traps hydrogen).
# ------------------------------------------------------------------------------------------
def desorption(S, mode_cooling, degas, prm, xp=np, mode_dwell=None):
    T = S["T_K"]
    arr = xp.exp(-(_g(prm["Qdes"], xp) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    open_heat = sigmoid((_g(prm["Tclose"], xp) - (T - 273.15)) / 10.0, xp)
    open_ = S["hotopen"] * xp.where(mode_cooling[..., None] > 0.5, 1.0, open_heat)
    if mode_dwell is not None:  # v4: load from the start of the dwell closes the pores there
        open_ = xp.where(mode_dwell[..., None] > 0.5, S["heating"], open_)
    theta = xp.sum(xp.exp(_g(prm["lnkdes"], xp)) * arr * open_ * S["dt"], axis=-1)
    h_init = degas * prm["hdg"] + (1.0 - degas) * 1.0
    return h_init * xp.exp(-theta)


def desorption_integrals(S, mode_cooling, prm, xp=np, mode_dwell=None):
    """v4. Desorption integrals over heating + dwell: (theta_closed, theta_open). theta_open keeps the pores open
    through heating and dwell (any load mode); theta_closed applies the load mode's pore closure (heating mode: smooth
    closure at Tclose; dwell mode: closed from the start of the dwell; cooling mode: open, so theta_closed ==
    theta_open exactly). Residual hydrogen = h_init * exp(-theta)."""
    T = S["T_K"]
    k = xp.exp(_g(prm["lnkdes"], xp)) * xp.exp(-(_g(prm["Qdes"], xp) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    open_heat = sigmoid((_g(prm["Tclose"], xp) - (T - 273.15)) / 10.0, xp)
    closed_mask = S["hotopen"] * xp.where(mode_cooling[..., None] > 0.5, 1.0, open_heat)
    if mode_dwell is not None:
        closed_mask = xp.where(mode_dwell[..., None] > 0.5, S["heating"], closed_mask)
    theta_open = xp.sum(k * S["hotopen"] * S["dt"], axis=-1)
    theta_closed = xp.sum(k * closed_mask * S["dt"], axis=-1)
    return theta_closed, theta_open


# ------------------------------------------------------------------------------------------
# Grain size (um): normal growth D^m = d0^m + k_g * int exp(-Qg/R(1/T-1/Tref)) dt  (m = 3),
# times an abnormal-growth factor (1 + A_ab * sigmoid((T_peak - T_ab)/15)) and anneal growth.
# ------------------------------------------------------------------------------------------
def grain_size(S, T_peak_C, anneal900, prm, m=3.0, xp=np):
    T = S["T_K"]
    arr = xp.exp(-(_g(prm["Qg"], xp) / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
    g = xp.sum(arr * S["dt"], axis=-1)
    # 900 C anneal, 120 min
    g_ann = anneal900 * 120.0 * xp.exp(-(prm["Qg"] / R_GAS) * (1.0 / 1173.15 - 1.0 / T_REF_K))
    Dm = prm["d0"] ** m + xp.exp(prm["lnkg"]) * (g + g_ann)
    D = Dm ** (1.0 / m)
    Tmax = xp.maximum(T_peak_C, xp.where(anneal900 > 0.5, 900.0, 0.0))
    return D * (1.0 + prm["Aab"] * sigmoid((Tmax - prm["Tab"]) / 15.0, xp))


def decomposition(T_peak_C, rate, prm, xp=np):
    """alpha-Fe / decomposition indicator in [0,1]; fast heating raises the local effective T."""
    Teff = T_peak_C + prm["cbeta"] * xp.log(rate / 10.0)
    return sigmoid((Teff - prm["Tdec"]) / prm["wdec"], xp)


def wetting(T_peak_C, mode_cooling, prm, xp=np):
    """GB-liquid continuity index W in [0,1] (H2): liquid spreading above T_w, squeezed out
    to triple junctions when load acts while liquid is present (heating mode)."""
    return sigmoid((T_peak_C - prm["Tw"]) / prm["ww"], xp) * (1.0 - prm["zeta"] * (1.0 - mode_cooling))


# ------------------------------------------------------------------------------------------
# Demagnetisation curve, energy product, temperature scaling
# ------------------------------------------------------------------------------------------
H_GRID = np.linspace(0.0, 1.0, 401)


def bhmax(Br, Hci, sq, shape=1.0, xp=np):
    """Parametric second-quadrant curve J(H) = Br (1 - (|H|/Hci)^q), q = ln 0.1 / ln(Hk/Hci),
    so J(-Hk) = 0.9 Br.  B = J + H (CGS).  Returns max(-H*B) in MGOe, clipped to Br^2/4.
    `shape` is a fitted multiplicative calibration against the RPI curves."""
    q = xp.log(0.1) / xp.log(xp.clip(sq, 1e-3, 0.999))
    h = H_GRID[None, :] * Hci[..., None] if xp.ndim(Hci) else H_GRID * Hci
    Brx = Br[..., None] if xp.ndim(Br) else Br
    Hx = Hci[..., None] if xp.ndim(Hci) else Hci
    qx = q[..., None] if xp.ndim(q) else q
    J = Brx * (1.0 - (h / Hx) ** qx)
    BH = h * xp.clip(J - h, 0.0, None)
    out = shape * xp.max(BH, axis=-1)
    return xp.minimum(out, Br ** 2 / 4.0)


def kuzmin_js(T_K, Js0, Tc=585.0, s=0.5, p=2.5, xp=np):
    """Kuz'min (2005) M(T)/M(0) = [1 - s tau^1.5 - (1-s) tau^p]^(1/3)."""
    tau = xp.clip(T_K / Tc, 0.0, 0.999)
    return Js0 * xp.clip(1.0 - s * tau ** 1.5 - (1.0 - s) * tau ** p, 1e-6, None) ** (1.0 / 3.0)
