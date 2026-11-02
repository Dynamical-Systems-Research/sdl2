"""Mechanism forward equations shared by the NumPyro models (xp = jax.numpy), the agent tool and the
world generator (xp = numpy). No jax import here.

Density mechanisms (relative density of an SPS body; unchanged from v3/v3.1):
  D1  diffusion-controlled (stress exponent n = 1), solid state
  D2  dislocation-creep-controlled (n in [2.5, 6])
  D3  liquid-phase-assisted (n = 1, rate x (1 + lam * sigmoid((T - TL)/wL)))
  D0  'other': D1 with a broad low-dimensional discrepancy (T slope, pressure-mode offset)

Coercivity mechanisms H (ln Hci relative to RPI's E route: 850 C, degassed, load on cooling, dense). v4: the H cores
are load-timing-neutral; why load during heating lowers coercivity is a separate indicator LT (below).
  H1  residual hydrogen (desorption kinetics with pores open through heating + dwell; no closure trapping here)
  H2  grain-boundary liquid continuity (wetting above Tw; the v3 squeeze-out term moved to LT = load)
  H3  alpha-Fe / decomposition (local overheating, fast ramps); parametrised by the penalty at sample A
  H4  grain size / Kronmueller Neff = nB ln(D/dB): Hci(D)/Hci(D_E) = 1 - (nB Js / H_KE) ln(D/D_E)
      (C4: a declared feedstock particle size acts on every mechanism through one empirical grain-size factor; H4's
      Kronmueller term stays at the 4 um feedstock's grain; see "C4 feedstock particle-size layer" below)
  H5  open porosity / oxidation of exposed grain boundaries
  H0  'other': generic historical contrasts + out-of-hull discrepancy

Load-timing indicator LT (v4, decision D5): ln Hci = H core - b_LT * Pi_LT(route), with b_LT the penalty at RPI's D
route (850 C, degassed, load in heating) relative to E, shared by the three alternatives, and
  Pi_k = [psi_k(x) - psi_k(x_E)] / [psi_k(x_D) - psi_k(x_E)],  psi_k(x) = 1 - exp(-kappa_k x / x_D)
so Pi = 1 exactly at D and 0 exactly at E (saturating; kappa_k = how saturated the damage is at D's dose). Doses x:
  trap     hydrogen trapped by early pore closure in excess of the open-pore residual:
           x = h0 [exp(-theta_closed) - exp(-theta_open)];  0 for load on cooling and for complete degassing (h0 = 0)
  load     load acting while the grain-boundary phase is liquid/soft (squeeze-out to triple junctions):
           x = integral (P / 50 MPa) sigmoid((T - Tsq) / wsq) dt over the whole schedule (persists when degassed)
  current  Joule overheating at necks when the loaded compact carries the pulsed current:
           x = path * m * exp(-(QI/R)(1/T_peak - 1/1123 K)), m = (rate/10)^nI with load in heating, 1 with load from
           the start of the dwell (holding current equated to a 10 C/min ramp's, declared), 0 for load on cooling or
           path = 0 (insulated compact)

C5 (declared, uncalibrated; see "C5 SEM/BSE observables and milled-feedstock density layer" at the end): model grain at
two stages of one piece (as sintered; after 900 C/2 h + 520 C/2 h), the grain-boundary continuity index W of a polished
BSE section, and a density layer for feedstock other than 4 um. None of these changes any function above.
"""

from __future__ import annotations

import numpy as np

from . import physics as ph

DENSITY_MECHS = ("D1", "D2", "D3", "D0")
MAG_MECHS = ("H1", "H2", "H3", "H4", "H5", "H0")
LT_LEVELS = ("trap", "load", "current")  # v4 load-timing indicator
E_RHO = 0.9738  # RPI E pellet relative density (7.42 / 7.62), used as the H5 reference
H_FLOOR = 1e-3  # kOe


def density_params(p: dict, mech: str) -> dict:
    out = dict(lnk=p["lnk"], Q=p["Q"], s=p["s"], p=1.0 + p["pm1"], phi0=p["phi0"], cR=p["cR"],
               lnkR=p["lnkR"], QR=p["QR"], phiR=p["phiR"],
               n=p["n"] if mech == "D2" else 1.0, d1=p["d1"], d2=p["d2"], rho_max=p["rho_max"])
    if mech == "D3":
        out.update(lam=p["lam"], TL=p["TL"], wL=p["wL"])
    else:
        out.update(lam=0.0, TL=700.0, wL=25.0)
    return out


def density(p: dict, mech: str, A: dict, lnk_offset=0.0, xp=np):
    prm = density_params(p, mech)
    prm["off"] = lnk_offset
    S = {k: xp.asarray(v) for k, v in A["S"].items()}
    return ph.densify(S, xp.asarray(A["cooling"]), xp.asarray(A["T_peak"]), xp.asarray(A["P"]), prm, xp=xp)


def _route_arrays(route):
    S = ph.stack([ph.schedule(route)])
    return dict(S=S, T_peak=np.array([route.T_peak_C]), rate=np.array([route.heating_rate_C_per_min]),
                cooling=np.array([1.0 if route.pressure_mode == "cooling" else 0.0]),
                mode_dwell=np.array([1.0 if route.pressure_mode == "dwell" else 0.0]),
                path=np.array([0.0 if route.current_path == "insulated" else 1.0]),
                degas=np.array([1.0]), hfac=np.array([1.0]), a900=np.array([0.0]), a520=np.array([0.0]))


# RPI's E (load on cooling) and D (load in heating) routes, slide 12 rows: 850 C, 10 C/min, 10 min, 50 MPa, degassed
E_ROUTE = ph.Route(T_peak_C=850.0, heating_rate_C_per_min=10.0, dwell_min=10.0, P_MPa=50.0, pressure_mode="cooling",
                   load_ramp_min=1.5, degas=1.0)
D_ROUTE = ph.Route(T_peak_C=850.0, heating_rate_C_per_min=10.0, dwell_min=10.0, P_MPa=50.0, pressure_mode="heating",
                   load_ramp_min=1.5, degas=1.0)
E_ARR = _route_arrays(E_ROUTE)
D_ARR = _route_arrays(D_ROUTE)
A_ROUTE = dict(T_peak=np.array([950.0]), rate=np.array([100.0]))  # RPI sample A (alpha-Fe marked)
H_MAX = 37.5  # kOe: smooth physical ceiling ~0.5 * mu0HA (Bance 2014: alpha <= 0.48 for ideal Dy-free grains)
T_D_K = 1123.15  # D's peak temperature (current-path dose reference)


# Magnetic data hull: the four independent specimens (A, C, D, E) in (T_peak C, relative density, load on cooling,
# degas, log10 heating rate), standardised by (100 C, 0.1, 1, 1, 1 decade). H0's out-of-hull discrepancy scales with
# the nearest-neighbour distance; it is 0 on every fitted row. Shared by the tool and the twin candidate.
MAG_HULL_POINTS = np.array([[950.0, 0.9882, 0.0, 0.0, 2.0],   # A: 950 C, 100 C/min, load in heating, no RPI degas
                            [850.0, 0.9777, 0.0, 0.0, 1.0],   # C
                            [850.0, 0.9829, 0.0, 1.0, 1.0],   # D
                            [850.0, 0.9738, 1.0, 1.0, 1.0]])  # E (and its anneals)
MAG_HULL_SCALES = np.array([100.0, 0.1, 1.0, 1.0, 1.0])


def mag_hull_distance(T_peak_C, rel_density, load_on_cooling, degas, rate_C_per_min):
    x = np.array([T_peak_C, rel_density, load_on_cooling, degas, np.log10(rate_C_per_min)], float)
    return float(np.sqrt((((MAG_HULL_POINTS - x) / MAG_HULL_SCALES) ** 2).sum(1)).min())


def cap_log_hci(lh, xp=np):
    """Smooth ceiling Hci <= H_MAX: Hci_c = Hci / (1 + (Hci/H_MAX)^4)^(1/4) (negligible below ~20 kOe)."""
    return lh - 0.25 * xp.logaddexp(0.0, 4.0 * (lh - np.log(H_MAX)))


def grain(p: dict, A: dict, xp=np):
    S = {k: xp.asarray(v) for k, v in A["S"].items()}
    if "fd_d0_factor" in A:  # C4 feedstock layer: this route's initial grain d0 x (d50 / 4 um)^beta_fd (E's unchanged)
        p = dict(p, d0=p["d0"] * xp.asarray(A["fd_d0_factor"]))
    return ph.grain_size(S, xp.asarray(A["T_peak"]), xp.asarray(A["a900"]), p, xp=xp)


# ------------------------------------------------------------------------------------------------------------------
# C4 feedstock particle-size layer (declared, uncalibrated; never fitted). d50: feedstock particle size (um) at the
# start of sintering; FEEDSTOCK_REF_UM = RPI's stated feedstock, at which every term below is exactly neutral (the tool
# does not call them there). Per-draw parameters fd_beta, fd_sg, fd_dox, fd_gox, fd_kH are prior draws: no record
# constrains them. Effects:
#   grain     d0 -> d0 (d50 / 4)^fd_beta in the grain-size law (growth terms unchanged): sintered grain D, against the
#             grain D_ref the same route and draw give from the 4 um feedstock (A without the hook)
#   coercivity every mechanism's Hci x softplus(1 - fd_sg ln(D / D_ref)) (an empirical relative log law; softplus floor
#             as in H4). H4's Kronmueller term is evaluated at D_ref, so a feedstock change enters it once, through
#             this factor, and H4 keeps its fitted response to temperature, dwell and anneal
#   oxidation ln Hci -= C (1 - exp(-fd_gox x / C)), x = max(0, fd_dox / d50 - 1) (specific surface in excess of that
#             of powder at the threshold size fd_dox), C = ln 10 (declared saturation); every mechanism
#   hydrogen  the milled powder holds m_H = 1 + fd_kH max(0, 1 - d50 / 4) times the current feedstock's hydrogen
#             before degassing; the route's degassing state is replaced by the current-feedstock state d' with the same
#             residual-hydrogen index, d' hdg + (1 - d') = m_H [d hdg + (1 - d)], which reaches H1, the trap
#             alternative and every generic degassing term; nothing changes with complete degassing (no hydrogen)
# ------------------------------------------------------------------------------------------------------------------
FEEDSTOCK_REF_UM = 4.0
FD_OX_CAP = float(np.log(10.0))  # ln kOe


def feedstock_d0_factor(p, d50, xp=np):
    """Factor on the initial grain size d0: (d50 / 4 um)^fd_beta."""
    return (d50 / FEEDSTOCK_REF_UM) ** p["fd_beta"]


def feedstock_grain_factor(p, D, D_ref, xp=np):
    """Empirical grain-size factor on Hci (every mechanism): softplus(1 - fd_sg ln(D / D_ref)), = 1 at D = D_ref."""
    r = 1.0 - p["fd_sg"] * xp.log(D / D_ref)
    return 0.02 * xp.logaddexp(0.0, r / 0.02)


def feedstock_oxidation_penalty(p, d50, xp=np):
    """Oxidation penalty on ln Hci (>= 0, saturating at ln 10): zero for d50 >= fd_dox."""
    x = xp.maximum(p["fd_dox"] / d50 - 1.0, 0.0)
    return FD_OX_CAP * -xp.expm1(-p["fd_gox"] * x / FD_OX_CAP)


def feedstock_hydrogen_factor(p, d50, xp=np):
    """Hydrogen of the milled powder before degassing relative to the current feedstock: 1 + fd_kH max(0, 1 - d50/4)."""
    return 1.0 + p["fd_kH"] * xp.maximum(1.0 - d50 / FEEDSTOCK_REF_UM, 0.0)


def feedstock_degas_equivalent(p, degas, d50, xp=np):
    """Current-feedstock degassing state with the milled powder's residual-hydrogen index:
    d' = (1 - m_H (1 - d (1 - hdg))) / (1 - hdg); d' = d when m_H = 1, d' < d when m_H > 1 (d' < 0: more hydrogen than
    non-degassed current powder)."""
    m = feedstock_hydrogen_factor(p, d50, xp)
    return (1.0 - m * (1.0 - degas * (1.0 - p["hdg"]))) / (1.0 - p["hdg"])


def decomp(p: dict, A: dict, xp=np):
    return ph.decomposition(xp.asarray(A["T_peak"]), xp.asarray(A["rate"]), p, xp=xp)


def _arr(A, key, like, default, xp):
    return xp.asarray(A[key]) if key in A else default * xp.ones_like(like)


# ------------------------------------------------------------------------------------------------------------------
# Load-timing doses and penalties (v4)
# ------------------------------------------------------------------------------------------------------------------
def _hydrogen(p, A, xp):
    """Initial hydrogen index h0 (1 = non-degassed powder, hdg = the fitted D/E degassed state, 0 = complete
    degassing via hfac = 0) and the desorption integrals (theta_closed, theta_open) of the route."""
    S = {k: xp.asarray(v) for k, v in A["S"].items()}
    cool = xp.asarray(A["cooling"])
    dw = _arr(A, "mode_dwell", cool, 0.0, xp)
    d = xp.asarray(A["degas"])
    h0 = (d * p["hdg"] + (1.0 - d)) * _arr(A, "hfac", cool, 1.0, xp)
    th_c, th_o = ph.desorption_integrals(S, cool, p, xp=xp, mode_dwell=dw)
    return h0, th_c, th_o


def hydrogen_open(p, A, xp=np):
    """Residual hydrogen index with the pores open through heating + dwell (load-timing-neutral; H1's core)."""
    h0, _, th_o = _hydrogen(p, A, xp)
    return h0 * xp.exp(-th_o)


LOG_TRAP_RATIO_MAX = 50.0  # the saturating penalty is flat long before r = e^50; the clamp only prevents overflow


def trap_ratio(p, A, xp=np):
    """x_trap(route) / x_trap(D), x_trap = h0 [exp(-theta_closed) - exp(-theta_open)] >= 0, computed in log space
    and clamped at e^50 (where every penalty is already saturated), so extreme kinetics never overflow. Exactly 0
    when nothing is trapped (load on cooling: theta_closed == theta_open; complete degassing: h0 = 0)."""
    h0, c, o = _hydrogen(p, A, xp)
    h0D, cD, oD = _hydrogen(p, D_ARR, xp)
    d = o - c
    pos = (d > 0) & (h0 > 0)
    sd = xp.where(pos, d, 1.0)    # safe values keep gradients finite in the branch that is not taken
    sh = xp.where(pos, h0, 1.0)
    lnum = xp.log(sh) - c + xp.log(-xp.expm1(-sd))
    lden = xp.log(h0D) - cD + xp.log(-xp.expm1(-xp.maximum(oD - cD, 1e-300)))
    return xp.where(pos, xp.exp(xp.minimum(lnum - lden, LOG_TRAP_RATIO_MAX)), 0.0)


def load_dose(p, A, xp=np):
    """Integral of (P / 50 MPa) x sigmoid((T - Tsq)/wsq) dt over the whole schedule (MPa-normalised minutes)."""
    S = {k: xp.asarray(v) for k, v in A["S"].items()}
    soft = ph.sigmoid((S["T_K"] - 273.15 - ph._g(p["Tsq"], xp)) / ph._g(p["wsq"], xp), xp)
    return xp.sum(S["P"] / ph.P_REF * soft * S["dt"], axis=-1)


def current_dose(p, A, xp=np):
    """Current-path dose relative to D (D = 1): path x m x exp(-(QI/R)(1/T_peak - 1/T_D))."""
    cool = xp.asarray(A["cooling"])
    dw = _arr(A, "mode_dwell", cool, 0.0, xp)
    path = _arr(A, "path", cool, 1.0, xp)
    heat = 1.0 - cool - dw
    m = heat * (xp.asarray(A["rate"]) / 10.0) ** p["nI"] + dw
    Tk = xp.asarray(A["T_peak"]) + 273.15
    return path * m * xp.exp(-(p["QI"] / ph.R_GAS) * (1.0 / Tk - 1.0 / T_D_K))


def _psi(r, kap, xp):
    return -xp.expm1(-kap * r)


def lt_penalties(p, A, xp=np):
    """Normalised load-timing penalty Pi_k(route) for k in LT_LEVELS (1 at D, 0 at E). Returns a dict."""
    out = {}
    out["trap"] = _psi(trap_ratio(p, A, xp), p["kap_tr"], xp) / _psi(1.0, p["kap_tr"], xp)
    xD = load_dose(p, D_ARR, xp)
    rE = load_dose(p, E_ARR, xp) / xD
    sE = _psi(rE, p["kap_ld"], xp)
    out["load"] = (_psi(load_dose(p, A, xp) / xD, p["kap_ld"], xp) - sE) / (_psi(1.0, p["kap_ld"], xp) - sE)
    out["current"] = _psi(current_dose(p, A, xp), p["kap_cu"], xp) / _psi(1.0, p["kap_cu"], xp)
    return out


# ------------------------------------------------------------------------------------------------------------------
# ln Hci
# ------------------------------------------------------------------------------------------------------------------
def log_hci(p: dict, mech: str, A: dict, rho, u_spec=0.0, hull_dist=0.0, z_hull=0.0, xp=np, lt="load"):
    """Body-level ln Hci (kOe) at room temperature, isotropic, for one parameter draw and load-timing label `lt`,
    with the smooth physical ceiling applied. u_spec: specimen effect (ln kOe). hull_dist/z_hull: H0 discrepancy."""
    return cap_log_hci(_log_hci_raw(p, mech, A, rho, u_spec, hull_dist, z_hull, xp, lt), xp)


def log_hci_all(p: dict, mech: str, A: dict, rho, u_spec=0.0, hull_dist=0.0, z_hull=0.0, xp=np):
    """Capped ln Hci for every load-timing label, stacked along a new leading axis in LT_LEVELS order."""
    core = _log_hci_core(p, mech, A, rho, u_spec, hull_dist, z_hull, xp)
    pen = lt_penalties(p, A, xp)
    return xp.stack([cap_log_hci(core - p["b_LT"] * pen[k], xp) for k in LT_LEVELS])


def _log_hci_raw(p, mech, A, rho, u_spec, hull_dist, z_hull, xp, lt):
    """Uncapped ln Hci for one load-timing label (a string, or an array of labels aligned with the draws)."""
    core = _log_hci_core(p, mech, A, rho, u_spec, hull_dist, z_hull, xp)
    pen = lt_penalties(p, A, xp)
    if isinstance(lt, str):
        return core - p["b_LT"] * pen[lt]
    lt = np.asarray(lt)
    sel = sum(np.where(lt == k, 1.0, 0.0).reshape(-1, *([1] * (np.ndim(pen[k]) - 1))) * pen[k] for k in LT_LEVELS)
    return core - p["b_LT"] * sel


def _log_hci_core(p, mech, A, rho, u_spec, hull_dist, z_hull, xp):
    """Load-timing-neutral ln Hci of mechanism `mech` (before the LT penalty and the ceiling)."""
    d = xp.asarray(A["degas"])
    hot = xp.where(xp.asarray(A["T_peak"]) >= 900.0, 1.0, 0.0)
    a1 = xp.asarray(A["a900"])
    a2 = xp.asarray(A["a520"])
    ann = p["b_a1"] * a1 + p["b_a2"] * a2
    if mech == "H4":
        # Kronmueller grain-size law relative to the E route (v3 form; see METHOD.md section 9 item 5a)
        D = grain(p, A, xp)
        DE = grain(p, E_ARR, xp)
        r = 1.0 - (p["nB"] * p["Js"] / p["HKE"]) * xp.log(D / DE)
        fl = 0.02
        ratio = fl * xp.logaddexp(0.0, r / fl)
        return p["muE"] + p["b_deg"] * (1 - d) + ann + xp.log(xp.maximum(ratio, 1e-6)) + u_spec
    if mech == "H1":
        h = hydrogen_open(p, A, xp)
        hE = hydrogen_open(p, E_ARR, xp)
        return p["muE"] - p["gH"] * (h - hE) + p["b_hot"] * hot + ann + u_spec
    if mech == "H2":
        W = ph.sigmoid((xp.asarray(A["T_peak"]) - p["Tw"]) / p["ww"], xp)
        WE = ph.sigmoid((850.0 - p["Tw"]) / p["ww"], xp)
        return p["muE"] + p["b_deg"] * (1 - d) + p["gL"] * (W - WE) + p["b_hot"] * hot + ann + u_spec
    if mech == "H3":
        # parametrised by gA, the decomposition penalty at RPI sample A: gF = gA / (f_A - f_E)
        f = decomp(p, A, xp)
        fE = ph.decomposition(850.0, 10.0, p, xp=xp)
        fA = decomp(p, A_ROUTE, xp)
        gF = p["gA"] / xp.maximum(fA - fE, 1e-3)
        return p["muE"] + p["b_deg"] * (1 - d) - gF * (f - fE) + ann + u_spec
    generic = p["muE"] + p["b_deg"] * (1 - d) + p["b_hot"] * hot + ann + u_spec
    if mech == "H5":
        po = xp.maximum(1.0 - rho - p["phic"], 0.0)
        poE = xp.maximum(1.0 - E_RHO - p["phic"], 0.0)
        return generic - p["gphi"] * (po - poE)
    if mech == "H0":
        return generic + p["tau0"] * hull_dist * z_hull
    raise ValueError(mech)


def remanence(p: dict, A: dict, rho, hci, xp=np):
    """Br (kG) at room temperature: Js * rho * <cos theta> * (1 + kappa) * R(Hci) * anneal factor.
    R(H) = 1 / (1 + (HR/H)^kR): remanence retention collapses when Hci falls below ~HR (A, C)."""
    R = 1.0 / (1.0 + (p["HR"] / xp.maximum(hci, H_FLOOR)) ** p["kR"])
    ann = xp.exp(p["ba1"] * xp.asarray(A["a900"]) + p["ba2"] * xp.asarray(A["a520"]))
    return p["Js"] * rho * xp.asarray(A["align"]) * (1.0 + p["kappa"]) * R * ann


def squareness_logit(p: dict, A: dict, xp=np):
    return p["eta0"] + p["eta_a1"] * xp.asarray(A["a900"]) + p["eta_a2"] * xp.asarray(A["a520"])


# ------------------------------------------------------------------------------------------------------------------
# C5 SEM/BSE observables and milled-feedstock density layer (declared, uncalibrated; never fitted by NUTS). Per-draw
# parameters c5_* are prior draws stored next to each posterior draw (c5_Tw, c5_ww only for coercivity mechanisms that
# have no fitted Tw, ww); the owner's importance reweighting on historical BSE readings (C5.0) is their only update.
#   grain     model grain D of the route's body at a stage: "as_sintered" (no anneal growth) or "after_900_520" (the
#             grain law's 900 C/2 h anneal growth; the 520 C step adds none). With a declared feedstock the C4 hook
#             d0 x (d50/4)^fd_beta applies to both stages.
#   W         continuity of the Nd-rich phase along grain boundaries in an as-sintered polished section, in [0, 1]:
#             W = sigmoid((T_peak - Tw)/ww) x clip(1 - c5_zeta Pi_LT(route), 0, 1) x exp(-c5_kox oxpen(d50)), with Pi_LT
#             the draw's load-timing penalty (1 at D's route, 0 at E's) and oxpen the C4 oxidation penalty (applied only
#             for a declared feedstock, as in C4). After 900 C/2 h + 520 C/2 h: W_ann = 1 - (1 - W)(1 - c5_rann), a
#             declared recovery fraction (neutral as sintered). A reading "spread" has probability
#             Phi((W_stage - c5_Wstar)/c5_tau).
#   density   for a declared feedstock only: ln-rates (densification and rearrangement, through the preparation-offset
#             slot) + [-c5_nD ln(d50/4) - c5_cox max(0, c5_dox/d50 - 1)], green relative density (1 - phi0) x
#             (d50/4)^c5_gam, and a symmetric discrepancy on the logit of rho / rho_max: + FD_DENSITY_LOGIT_SD
#             |log2(d50/4)| c5_zfd, c5_zfd ~ N(0, 1) per draw (agglomeration, oxide and packing are not modelled; it
#             widens the band away from 4 um). Not evaluated at 4 um (the tool does not call it there).
# ------------------------------------------------------------------------------------------------------------------
SEM_STAGES = ("as_sintered", "after_900_520")
STAGE_ALIASES = {"900+520": "after_900_520"}  # accepted on input
GRAIN_FIELD_LOG_SD = 0.15  # declared: field sampling of a reported mean grain size (log-sd)


def grain_stage(p, A, stage, d0_factor=None, xp=np):
    """Model grain (um) of the route's body at `stage`, independent of the route's own anneal key."""
    stage = STAGE_ALIASES.get(stage, stage)
    if stage not in SEM_STAGES:
        raise ValueError(f"stage must be one of {SEM_STAGES}")
    a = 1.0 if stage == "after_900_520" else 0.0
    As = dict(A, a900=np.array([a]), a520=np.array([a]))
    if d0_factor is not None:
        As["fd_d0_factor"] = d0_factor
    return grain(p, As, xp)


def bse_liquid(p, T_peak, xp=np):
    """Liquid term sigmoid((T_peak - Tw)/ww): H2's fitted Tw, ww where the draw has them, else the stored prior draws."""
    Tw = p["Tw"] if "Tw" in p else p["c5_Tw"]
    ww = p["ww"] if "ww" in p else p["c5_ww"]
    return ph.sigmoid((xp.asarray(T_peak) - Tw) / ww, xp)


def bse_continuity(p, A, pen, d50=None, xp=np):
    """Continuity index W in [0, 1]. pen: the draw's load-timing penalty Pi_LT at this route (aligned with p).
    d50: a declared feedstock size (the oxide term is not evaluated without it)."""
    W = bse_liquid(p, A["T_peak"], xp) * xp.clip(1.0 - p["c5_zeta"] * pen, 0.0, 1.0)
    if d50 is not None:
        W = W * xp.exp(-p["c5_kox"] * feedstock_oxidation_penalty(p, d50, xp))
    return W


def bse_stage_continuity(p, W, stage, xp=np):
    """W at a stage: unchanged as sintered; 1 - (1 - W)(1 - c5_rann) after 900 C/2 h + 520 C/2 h (declared recovery)."""
    stage = STAGE_ALIASES.get(stage, stage)
    if stage == "as_sintered":
        return W
    if stage == "after_900_520":
        return 1.0 - (1.0 - W) * (1.0 - p["c5_rann"])
    raise ValueError(f"stage must be one of {SEM_STAGES}")


def std_normal_sf(z):
    """Upper tail 1 - Phi(z), accurate in both tails (numpy has no erfc)."""
    import math
    z = np.asarray(z, float)
    return 0.5 * np.frompyfunc(math.erfc, 1, 1)(z / math.sqrt(2.0)).astype(float)


def bse_p_spread(p, W):
    """P(reading 'spread' | W) = Phi((W - c5_Wstar) / c5_tau)."""
    return std_normal_sf(-(np.asarray(W) - p["c5_Wstar"]) / p["c5_tau"])


def feedstock_density_offset(p, d50, xp=np):
    """ln-rate offset of the declared density layer: -c5_nD ln(d50/4) - c5_cox max(0, c5_dox/d50 - 1)."""
    x = xp.maximum(p["c5_dox"] / d50 - 1.0, 0.0)
    return -p["c5_nD"] * xp.log(d50 / FEEDSTOCK_REF_UM) - p["c5_cox"] * x


# declared: sd of the density discrepancy per halving or doubling of d50 (logit of rho / rho_max). Agglomeration, oxide
# and packing of fine powder are not modelled and can lower as well as raise density; 1.25 makes the 90 % band at 1 and
# 2 um at least as wide as at 4 um on the same route (850 C panel route, a new preparation). No record constrains it.
FD_DENSITY_LOGIT_SD = 1.25


def feedstock_density_discrepancy(p, rho, d50, xp=np):
    """Relative density with the declared milled-feedstock discrepancy: logit(rho / rho_max) shifted by
    FD_DENSITY_LOGIT_SD |log2(d50 / 4)| c5_zfd (zero at 4 um; lower and higher density alike)."""
    x = xp.clip(rho / p["rho_max"], 1e-9, 1.0 - 1e-9)
    z = xp.log(x / (1.0 - x)) + FD_DENSITY_LOGIT_SD * abs(np.log2(d50 / FEEDSTOCK_REF_UM)) * p["c5_zfd"]
    return p["rho_max"] / (1.0 + xp.exp(-z))


def feedstock_green_phi0(p, d50, xp=np):
    """Loose-powder porosity with green relative density (1 - phi0) x (d50/4)^c5_gam."""
    return 1.0 - (1.0 - p["phi0"]) * (d50 / FEEDSTOCK_REF_UM) ** p["c5_gam"]
