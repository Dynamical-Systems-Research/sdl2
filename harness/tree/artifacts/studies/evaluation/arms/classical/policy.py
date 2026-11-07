"""SDL-2 classical policy arms: design of experiments (`doe`) and Bayesian optimisation (`bo`, with the supplied
model's prior).

    python/bin/python -B arms/classical/policy.py STATE.json      # prints one JSON action on stdout

Information. The policy reads only STATE.json: the contract's numbers, the reference body's observations, the agent-
visible returns of its own requests, the economy ledger, its own action history, and (bo only) the disclosed
supplied-model prior file named in `prior`. It imports numpy, scipy and ndfeb_sps.economy and nothing else from the
study, and opens no file other than STATE.json and the prior.

Determinism. Every action is a pure function of STATE.json (the seed included). The investigation is a fixed sequence
of rounds; each round's requests carry the round tag in their request_id (`<ARM>-<TAG>-<k>`), and a round's plan is
computed from the record of the rounds before it only, so recomputing it at a later call gives the same plan. The
round in progress is the last one with a request in `history`; the next action is its first request not yet attempted,
else the next round's, else the claim.

Shared statistical model (constants below):
  ln Hci        screens and loops; route features (+ the supplied model's per-route ln Hci median as a fixed offset in
                bo); preparation and body random effects; reading error
  ln Br         loops directly; a density reading as ln(density) + kappa, kappa = ln(Br / density) estimated from the
                policy's own bodies with both readings at one stage; same features, effects and offsets
  ln s          s = (BH)max / (Br^2 / 4), the isotropic squareness factor, linear in ln(Hci / Br), capped at 1
doe weights three ln Hci (preparation, body) variance pairs by their marginal likelihood (VAR_GRID); bo uses the earlier
study's pair. A body meets the requirement when Hci >= 10 kOe and (BH)max >= 11 MGOe. For a fresh preparation of three bodies,
p3 = E_prep[P(body meets | prep)^3], integrated by Gauss-Hermite quadrature over the preparation effects; a route's
P(meets) is the posterior probability that p3 >= 0.95, and P(any declared route meets) the posterior probability that
some route in the arm's candidate set has p3 >= 0.95.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr

from ndfeb_sps import economy as economy

# ------------------------------------------------------------------------------------------------ frozen configuration
# ln Hci (preparation, body) variance pairs weighted by marginal likelihood on the record: small scatter, the earlier
# study's values, and large scatter (the contract does not state this lot's scatter)
VAR_GRID = [[0.03, 0.03], [0.10, 0.07], [0.15, 0.20]]
CONFIG = {
    "doe": {"tau_restrict": 0.50, "tau_replicate": 0.10, "n_theta": 48, "var_grid": VAR_GRID,
            "anneal_contrast": ["900_1h+520", "900_4h+520"], "contrast_bodies": 2,
            "steepest_ascent_bodies": 2, "loops_leading": 2,
            # credits held back after each stage: D1/D2 keep the D3 contrast (2 bodies x 2 steps x 20 = 80) + D4 loops
            # (76) + build (171); D3 keeps D4 + build; D4 keeps the build (new preparation 15, two bodies with anneals
            # and density ~60, two loop extensions 76: 171)
            "reserve_d1": 327, "reserve_d3": 247, "reserve_build": 171},
    # bo's development variants were "var_grid": VAR_GRID and "start_design": True with
    # "explore_rounds": [3, 2]; neither is frozen, so both stay off here
    "bo": {"tau_restrict": 0.50, "tau_replicate": 0.10, "n_theta": 48, "explore_rounds": [4, 3], "loops_leading": 2,
           "anneal_margin": 0.02, "start_design": False,
           # exploration keeps the L loops (76) + the build (new preparation 15, two bodies with anneals and density
           # ~60, two loop extensions 76: 171)
           "reserve_explore": 247, "reserve_build": 171},
}
REQUIREMENT = {"hci_kOe": 10.0, "bhmax_MGOe": 11.0, "bodies": 3, "p_all": 0.95}
LOT = "SDL2-LOT"
PIECE = "P1"
DENSITY_METHOD = "archimedes_saturated"   # reads bulk density without bias (CONTRACT.md); same price as the others
PASS_SUPPORT = 0.95                       # a loop "meets" when P(true state meets | reading) >= 0.95 (2 % reading model)
STAGES = ("as_sintered", "after_first_anneal", "after_second_anneal")

# Variance components. Structure ported from the earlier study (cb/model.py): preparation (batch) and body effects on
# ln Hci and on ln Br, a reading error, a squareness residual. Values:
#   ln Hci batch 0.10, body 0.07          earlier study (public reference-lab sigma_lnh_batch, sigma_z_body); the
#                                          contract states this lot's scatter is not stated
#   ln Br batch 0.02, body 0.012          builder assumption: the earlier study's density body 0.020 g/cm3 and
#                                          ln(Br/rho) body 0.010 combine to ~0.012; the batch term is widened from its
#                                          0.010 g/cm3 because preparations here also differ in feedstock
#   reading 0.02 (loops), 0.03 (screens)  contract: independent 2 % log reading errors on Hci and Br; the screen's error
#                                          is not stated (builder assumption, wider)
#   ln s residual 0.03                    earlier study's SV
#   kappa = ln(Br/rho): prior N(0, 0.06), route scatter 0.03 (builder assumption from textbook physics: isotropic
#                                          remanence is about half of Js ~ 16 kG at full density ~ 7.6 g/cm3, so Br in
#                                          kG is close to the density in g/cm3; learned from the policy's own pairs)
H_BATCH, H_BODY = 0.10, 0.07
B_BATCH, B_BODY = 0.02, 0.012
LOOP_SD, SCREEN_SD = 0.02, 0.03
S_RESID = 0.03
KAPPA_PRIOR, KAPPA_SD = (0.0, 0.06), 0.03
X0 = 0.3                                   # centring of ln(Hci/Br) in the squareness relation
LN10, LN11, LN4 = math.log(10.0), math.log(11.0), math.log(4.0)

# Route features (main effects plus the interactions the process justifies): coded controls; t = peak temperature,
# d = feedstock fineness log2(4/d50), a9/ah/a5 = 900 C anneal, its hold (log2 h, centred at 2 h) and the 520 C step.
# t*a9: the 900 C anneal's densification depends on the as-sintered density (peak temperature); d*t: finer powder
# sinters at a lower peak; d*a9: grain growth in the 900 C hold depends on the starting grain; t*a5: the 520 C response
# at the sintered state.
FEATURES = ["int", "none", "d1", "d2", "t1", "t2", "h900", "r100", "p40", "heat", "dwell", "c700", "ins",
            "a9", "ah", "a5", "t1a9", "d1t1", "d1a9", "t1a5"]
# Prior sds (means 0 except the intercepts). doe: weakly informative around the lot's reference body (its readings
# enter as data); bo: the discrepancy from the supplied model's prediction, so narrower.
PRIOR_SD = {
    "doe": {"H": {"int": 0.6, "none": 1.5, "d1": 0.4, "d2": 0.3, "t1": 0.4, "t2": 0.3, "h900": 0.3, "r100": 0.3,
                  "p40": 0.3, "heat": 0.4, "dwell": 0.4, "c700": 0.4, "ins": 0.3, "a9": 0.3, "ah": 0.15, "a5": 0.3,
                  "t1a9": 0.2, "d1t1": 0.2, "d1a9": 0.2, "t1a5": 0.2},
            "B": {"int": 0.15, "none": 0.2, "d1": 0.1, "d2": 0.05, "t1": 0.08, "t2": 0.05, "h900": 0.05, "r100": 0.05,
                  "p40": 0.05, "heat": 0.08, "dwell": 0.08, "c700": 0.08, "ins": 0.05, "a9": 0.08, "ah": 0.04,
                  "a5": 0.05, "t1a9": 0.05, "d1t1": 0.05, "d1a9": 0.05, "t1a5": 0.05}},
    "bo": {"H": {"int": 0.30, "none": 0.5, "d1": 0.2, "d2": 0.15, "t1": 0.2, "t2": 0.15, "h900": 0.15, "r100": 0.15,
                 "p40": 0.15, "heat": 0.2, "dwell": 0.2, "c700": 0.2, "ins": 0.15, "a9": 0.15, "ah": 0.08, "a5": 0.15,
                 "t1a9": 0.1, "d1t1": 0.1, "d1a9": 0.1, "t1a5": 0.1},
           "B": {"int": 0.08, "none": 0.1, "d1": 0.05, "d2": 0.03, "t1": 0.04, "t2": 0.03, "h900": 0.03, "r100": 0.03,
                 "p40": 0.03, "heat": 0.04, "dwell": 0.04, "c700": 0.04, "ins": 0.03, "a9": 0.04, "ah": 0.02,
                 "a5": 0.03, "t1a9": 0.03, "d1t1": 0.03, "d1a9": 0.03, "t1a5": 0.03}},
}
DOE_INTERCEPT = {"H": math.log(8.0), "B": math.log(7.2)}
S_PRIOR_DOE = {"sa": (math.log(0.78), 0.12), "sg": (0.30, 0.20)}
S_PRIOR_SD_BO = {"sa": 0.06, "sg": 0.12}


# ------------------------------------------------------------------------------------------------ declared routes
ROUTES = economy.declared_routes()
RIDS = [economy.route_id(r) for r in ROUTES]
INDEX = {rid: i for i, rid in enumerate(RIDS)}
R = len(ROUTES)


def _col(key):
    return np.array([r[key] for r in ROUTES], dtype=object)


def route_features(routes):
    """Design matrix (n, len(FEATURES)) for a list of declared routes."""
    rows = []
    for r in routes:
        d1 = math.log2(4.0 / float(r["feedstock_d50_um"]))
        t1 = (float(r["temperature_c"]) - 825.0) / 75.0
        an = r["anneal"]
        a9 = 0.0 if an == "none" else 1.0
        ah = 0.0 if an == "none" else math.log2(float(an.split("_")[1].split("h")[0])) - 1.0
        a5 = 1.0 if an.endswith("+520") else 0.0
        load = (r["pressure_schedule"], r["pressure_full_at_c"])
        rows.append([1.0, float(r["preparation"] == "none"), d1, d1 * d1, t1, t1 * t1,
                     float(r["hold_time_s"] == 900.0), float(r["heating_rate_c_min"] == 100.0),
                     float(r["pressure_mpa"] == 40.0), float(load[0] == "on_heating"), float(load[0] == "on_dwell"),
                     float(load == ("on_cooling", 700.0)), float(r["current_path"] == "insulated"),
                     a9, ah, a5, t1 * a9, d1 * t1, d1 * a9, t1 * a5])
    return np.array(rows, float)


X_ALL = route_features(ROUTES)
BASE_OF = np.array([economy.route_id(dict(r, anneal="none")) for r in ROUTES])
ANNEALS = list(economy.DECLARED["anneal"])
STAGE_OF_ANNEAL = {a: STAGES[len(economy.ANNEAL_PROGRAMS[a])] for a in ANNEALS}


def with_anneal(route, anneal):
    return INDEX[economy.route_id(dict(route, anneal=anneal))]


# ------------------------------------------------------------------------------------------------ the record
@dataclass
class Acq:
    sid: str
    stage: str
    fsid: str
    request_id: str
    route: int | None = None
    prep: str | None = None
    hci: float | None = None
    hci_kind: str | None = None        # "loop" or "screen"
    br: float | None = None
    bh: float | None = None
    rho: float | None = None
    rho_sd: float | None = None
    loop_ids: tuple = ()


class Record:
    """The policy's view of the returns: bodies (economy.ledger) and acquisitions (one per specimen, stage and
    field_set_id), each mapped to the declared route its body was at that stage."""

    def __init__(self, returns):
        self.returns = list(returns)
        self.entries = [{"request_id": r["request_id"], "request": r["request"], "delivered": True} for r in self.returns]
        self.ledger = economy.ledger(self.entries)
        self.bodies = self.ledger["bodies"]
        self.preps = self.ledger["preparations"]
        self.cycle, self.intended = {}, {}
        for r in self.returns:
            if r["request"].get("operation", "manufacture") == "manufacture":
                for s in r["request"].get("specimens", []):
                    self.cycle[s["specimen_id"]] = s["cycle_id"]
                    tail = s["cycle_id"].rsplit("-R", 1)
                    if len(tail) == 2 and tail[1].isdigit() and int(tail[1]) < R:
                        self.intended[s["specimen_id"]] = int(tail[1])   # the route the policy made it for
        acqs = {}
        for r in self.returns:
            for o in (r.get("delivery") or {}).get("observations", []):
                sid = o.get("specimen_id")
                stage = o.get("process_stage") or o.get("measurement_stage")
                key = (sid, stage, o.get("acquisition_id"))
                a = acqs.get(key)
                if a is None:
                    a = acqs[key] = Acq(sid, stage, o.get("acquisition_id"), r["request_id"])
                q, v = o.get("quantity"), o.get("value")
                lr = o.get("loop_reading") or {}
                if q == "density_g_cm3" and isinstance(v, (int, float)):
                    if o.get("density_method") == DENSITY_METHOD:
                        a.rho = float(v)
                        a.rho_sd = float((o.get("uncertainty") or {}).get("reading_sd_g_cm3") or 0.05)
                elif q == "hci_screen_kOe" and isinstance(v, (int, float)) and a.hci_kind != "loop":
                    a.hci, a.hci_kind = float(v), "screen"
                elif q in ("hci_kOe", "br_kGs", "bhmax_MGOe", "demagnetization_curve"):
                    if isinstance(lr.get("hci_kOe"), (int, float)):
                        a.hci, a.hci_kind = float(lr["hci_kOe"]), "loop"
                    if isinstance(lr.get("br_kGs"), (int, float)):
                        a.br = float(lr["br_kGs"])
                    if isinstance(lr.get("bhmax_MGOe"), (int, float)):
                        a.bh = float(lr["bhmax_MGOe"])
                    a.loop_ids = tuple(sorted(set(a.loop_ids) | {o.get("observation_id")}))
        for a in acqs.values():
            b = self.bodies.get(a.sid)
            if b is None or a.stage not in STAGES:
                continue
            a.prep = b["preparation_id"]
            n = STAGES.index(a.stage)
            route = economy.declared_route({**b, "stage": a.stage, "steps": b["steps"][:n]}, self.preps)
            a.route = INDEX[economy.route_id(route)] if route else None
        self.acqs = list(acqs.values())

    def own(self, sid):
        b = self.bodies.get(sid)
        return b is not None and not b["free"]

    @staticmethod
    def own_acq(a):
        """An acquisition from the policy's own request (a loop on the reference body in an own request counts)."""
        return a.request_id not in economy.FREE_REQUESTS

    def own_bodies(self):
        return [sid for sid, b in self.bodies.items() if not b["free"]]

    def current_route(self, sid):
        b = self.bodies[sid]
        route = economy.declared_route(b, self.preps)
        return INDEX[economy.route_id(route)] if route else None

    def acq_at(self, sid, stage):
        return [a for a in self.acqs if a.sid == sid and a.stage == stage]


def p_pass_reading(hci, bh, sg=0.3):
    """P(the true state meets both limits | one loop reading) under the declared 2 % log reading errors on Hci and Br;
    (BH)max moves with both (d ln BH = sg d ln H + (2 - sg) d ln Br). Product of the two margins (conservative)."""
    sd_bh = LOOP_SD * math.sqrt(sg * sg + (2 - sg) ** 2)
    return float(ndtr((math.log(hci) - LN10) / LOOP_SD) * ndtr((math.log(bh) - LN11) / sd_bh))


def loop_summary(rec):
    """Own full loops (with (BH)max) per declared route at its own stage: {route: [(sid, p_pass, acq)]}."""
    out = {}
    for a in rec.acqs:
        if a.route is None or a.bh is None or a.hci is None or not rec.own_acq(a):
            continue
        out.setdefault(a.route, []).append((a.sid, p_pass_reading(a.hci, a.bh), a))
    return out


def verified_routes(rec):
    """Routes with own loops on at least two bodies, every loop on the route meeting both limits."""
    out = {}
    for r, loops in loop_summary(rec).items():
        if len({s for s, _, _ in loops}) >= 2 and all(p >= PASS_SUPPORT for _, p, _ in loops):
            out[r] = loops
    return out


# ------------------------------------------------------------------------------------------------ the prior (bo)
def load_prior(spec):
    """The supplied model's per-route medians for both magnetic calculations (`prior` in STATE.json: a file, a
    glob, a directory or a list). Returns {path: {"H", "B", "S": arrays over ROUTES}} and the fraction of routes found.
    Missing routes (a file still filling) take the mean of the found routes with the same preparation and feedstock."""
    paths = []
    for s in (spec if isinstance(spec, list) else [spec]):
        if not s:
            continue
        s = str(s)
        if os.path.isdir(s):
            paths += sorted(glob.glob(os.path.join(s, "supplied_ranking_*.jsonl")))
        elif any(ch in s for ch in "*?["):
            paths += sorted(glob.glob(s))
        elif os.path.exists(s):
            paths.append(s)
    out = {p: {k: np.full(R, np.nan) for k in ("H", "B", "S")} for p in ("link", "c5")}
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                i = INDEX.get(row.get("id"))
                if i is None:
                    continue
                for p in ("link", "c5"):
                    m = (row.get(p) or {}).get("medians") or {}
                    h, b, bh = m.get("Hci_kOe"), m.get("Br_kGs"), m.get("BHmax_MGOe")
                    if all(isinstance(x, (int, float)) and x > 0 for x in (h, b, bh)):
                        out[p]["H"][i], out[p]["B"][i] = math.log(h), math.log(b)
                        out[p]["S"][i] = min(math.log(bh / (b * b / 4.0)), 0.0)
    found = float(np.mean(np.isfinite(out["link"]["H"])))
    if found == 0.0:
        return None, 0.0
    group = np.array([f"{r['preparation']}|{r['feedstock_d50_um']}" for r in ROUTES])
    for p in out:
        for k, v in out[p].items():
            miss = ~np.isfinite(v)
            for g in np.unique(group[miss]):
                sel = group == g
                ok = sel & np.isfinite(v)
                v[sel & miss] = np.nanmean(v[ok]) if ok.any() else np.nanmean(v)
    return out, found


def squareness_prior(prior, path):
    """The squareness relation implied by the supplied model's own medians (fitted-preparation routes with 3-30 kOe):
    least squares of ln s on ln(Hci/Br) - X0, the prior mean of the bo arm's squareness fit for that path."""
    H, B, S = prior[path]["H"], prior[path]["B"], prior[path]["S"]
    ok = np.isfinite(S) & (H > math.log(3.0)) & (H < math.log(30.0)) & (S < -0.01)
    if ok.sum() < 20:
        return {"sa": S_PRIOR_DOE["sa"], "sg": S_PRIOR_DOE["sg"]}
    x = H[ok] - B[ok] - X0
    A = np.column_stack([np.ones(ok.sum()), x])
    coef = np.linalg.lstsq(A, S[ok], rcond=None)[0]
    return {"sa": (float(coef[0]), S_PRIOR_SD_BO["sa"]), "sg": (float(coef[1]), S_PRIOR_SD_BO["sg"])}


# ------------------------------------------------------------------------------------------------ Gaussian fits
def gls(X, y, s_meas, batch, body, s_batch, s_body, m0, sd0):
    """y = X theta + batch + body + reading error, theta ~ N(m0, diag(sd0^2)), variance components known. Returns the
    posterior mean and covariance of theta and the log marginal likelihood of y."""
    m0, S0 = np.asarray(m0, float), np.diag(np.asarray(sd0, float) ** 2)
    if len(y) == 0:
        return m0, S0, 0.0
    X, y = np.asarray(X, float), np.asarray(y, float)
    bt, bd = np.asarray(batch), np.asarray(body)
    Om = (np.diag(np.asarray(s_meas, float) ** 2) + s_batch ** 2 * (bt[:, None] == bt[None, :])
          + s_body ** 2 * (bd[:, None] == bd[None, :]))
    Oi = np.linalg.inv(Om)
    cov = np.linalg.inv(np.linalg.inv(S0) + X.T @ Oi @ X)
    cov = 0.5 * (cov + cov.T)
    mean = cov @ (np.linalg.solve(S0, m0) + X.T @ Oi @ y)
    V = X @ S0 @ X.T + Om
    r = y - X @ m0
    sign, logdet = np.linalg.slogdet(V)
    ll = -0.5 * (logdet + r @ np.linalg.solve(V, r) + len(y) * math.log(2 * math.pi))
    return mean, cov, float(ll)


class Posterior:
    """The fitted model for one arm: one component per magnetic-path prior (bo: link and c5 offsets; doe: one
    component without offsets) and, when var_grid is given, per (ln Hci batch, body) variance pair; components are
    weighted by their marginal likelihood on the record (equal prior weights)."""

    def __init__(self, arm, rec, prior=None, var_grid=None, sd_override=None):
        self.arm = arm
        self.comps = []
        H_obs, B_obs, S_obs = self._observations(rec)
        paths = [None] if prior is None else ["link", "c5"]
        sd = PRIOR_SD[arm if prior is not None else "doe"]   # bo without its prior file falls back to doe's priors
        sd = {k: dict(v, **((sd_override or {}).get(k) or {})) for k, v in sd.items()}
        for p, (hb, hv) in [(p, v) for p in paths for v in (var_grid or [(H_BATCH, H_BODY)])]:
            offH = np.zeros(R) if p is None else prior[p]["H"]
            offB = np.zeros(R) if p is None else prior[p]["B"]
            mH = np.zeros(len(FEATURES))
            mB = np.zeros(len(FEATURES))
            if p is None:
                mH[0], mB[0] = DOE_INTERCEPT["H"], DOE_INTERCEPT["B"]
            sp = S_PRIOR_DOE if p is None else squareness_prior(prior, p)
            fitH = self._fit(H_obs, offH, mH, [sd["H"][f] for f in FEATURES], hb, hv)
            fitB = self._fit(B_obs, offB, mB, [sd["B"][f] for f in FEATURES], B_BATCH, B_BODY)
            Xs = [[1.0, x] for x, _ in S_obs]
            fitS = gls(Xs, [y for _, y in S_obs], [S_RESID] * len(S_obs), list(range(len(S_obs))),
                       list(range(len(S_obs))), 0.0, 0.0, [sp["sa"][0], sp["sg"][0]], [sp["sa"][1], sp["sg"][1]])
            self.comps.append({"path": p, "hb": hb, "hv": hv, "offH": offH, "offB": offB, "H": fitH, "B": fitB,
                               "S": fitS, "ll": fitH[2] + fitB[2] + fitS[2]})
        ll = np.array([c["ll"] for c in self.comps])
        w = np.exp(ll - ll.max())
        self.weights = w / w.sum()

    @staticmethod
    def _fit(obs, off, m0, sd0, s_batch, s_body):
        X = [X_ALL[r] for r, *_ in obs]
        y = [v - off[r] for r, v, *_ in obs]
        return gls(X, y, [s for _, _, s, _, _ in obs], [b for *_, b, _ in obs], [b for *_, b in obs],
                   s_batch, s_body, m0, sd0)

    @staticmethod
    def _observations(rec):
        """(route, value, sd, batch, body) rows for ln Hci and ln Br; (x, y) rows for ln s."""
        H, B, S, kap = [], [], [], []
        for a in rec.acqs:
            if a.route is None:
                continue
            if a.hci:
                H.append((a.route, math.log(a.hci), LOOP_SD if a.hci_kind == "loop" else SCREEN_SD, a.prep, a.sid))
            if a.br:
                B.append((a.route, math.log(a.br), LOOP_SD, a.prep, a.sid))
                if a.rho:
                    kap.append(math.log(a.br / a.rho))
                if a.bh and a.hci:
                    S.append((math.log(a.hci / a.br) - X0, math.log(a.bh / (a.br * a.br / 4.0))))
        # kappa = ln(Br / density): normal-normal update from the own pairs
        prec = 1 / KAPPA_PRIOR[1] ** 2 + len(kap) / KAPPA_SD ** 2
        k_mean = (KAPPA_PRIOR[0] / KAPPA_PRIOR[1] ** 2 + sum(kap) / KAPPA_SD ** 2) / prec
        k_var = 1 / prec
        for a in rec.acqs:
            if a.route is None or a.br or not a.rho:
                continue
            sd = math.sqrt((a.rho_sd / a.rho) ** 2 + k_var + KAPPA_SD ** 2)
            B.append((a.route, math.log(a.rho) + k_mean, sd, a.prep, a.sid))
        return H, B, S

    def draws(self, n, rng):
        """n posterior draws: component (stratified by weight; one random component for a single Thompson draw),
        theta_H, theta_B, (sa, sg)."""
        cum = np.cumsum(self.weights)
        u = rng.random(1) if n == 1 else (np.arange(n) + 0.5) / n
        comp = np.searchsorted(cum, u)
        comp = np.minimum(comp, len(self.comps) - 1)
        out = {"comp": comp, "tH": np.zeros((n, len(FEATURES))), "tB": np.zeros((n, len(FEATURES))),
               "S": np.zeros((n, 2)), "hb": np.array([self.comps[c]["hb"] for c in comp]),
               "hv": np.array([self.comps[c]["hv"] for c in comp])}
        for k, c in enumerate(self.comps):
            idx = np.where(comp == k)[0]
            if len(idx) == 0:
                continue
            for key, fit in (("tH", c["H"]), ("tB", c["B"]), ("S", c["S"])):
                L = np.linalg.cholesky(fit[1] + 1e-12 * np.eye(len(fit[0])))
                out[key][idx] = fit[0][None, :] + rng.standard_normal((len(idx), len(fit[0]))) @ L.T
        return out

    def mu(self, dr, idx=None):
        """Route means of ln Hci and ln Br, (routes, draws)."""
        X = X_ALL if idx is None else X_ALL[idx]
        muH = X @ dr["tH"].T
        muB = X @ dr["tB"].T
        for k, c in enumerate(self.comps):
            sel = dr["comp"] == k
            if sel.any():
                offH = c["offH"] if idx is None else c["offH"][idx]
                offB = c["offB"] if idx is None else c["offB"][idx]
                muH[:, sel] += offH[:, None]
                muB[:, sel] += offB[:, None]
        return muH, muB

    def p3(self, dr, idx=None):
        muH, muB = self.mu(dr, idx)
        return p3_table(muH, muB, dr["S"][:, 0], dr["S"][:, 1], dr["hb"], dr["hv"])

    def point(self, idx):
        """Posterior-mean predictions (Hci, Br, (BH)max) for routes idx under the weighted components."""
        out = []
        for i in idx:
            h = b = s = 0.0
            for w, c in zip(self.weights, self.comps):
                lh = c["offH"][i] + X_ALL[i] @ c["H"][0]
                lb = c["offB"][i] + X_ALL[i] @ c["B"][0]
                ls = min(c["S"][0][0] + c["S"][0][1] * (lh - lb - X0), 0.0)
                h, b, s = h + w * lh, b + w * lb, s + w * ls
            out.append((math.exp(h), math.exp(b), math.exp(s + 2 * b - LN4)))
        return out


_XU, _WU = hermegauss(12)
_XB, _WB = hermegauss(6)
_W = (_WU[:, None] * _WB[None, :]) / (_WU.sum() * _WB.sum())


def p3_table(muH, muB, sa, sg, hb, hv, chunk=600):
    """P(all three bodies of a fresh preparation meet), (routes, draws): Gauss-Hermite over the preparation effects on
    ln Hci (12 nodes) and ln Br (6 nodes); given the preparation, a body meets with P(Hci >= 10) * P(BH >= 11), its
    (BH)max lognormal through the squareness relation (the positive Hci-(BH)max correlation is dropped: conservative)."""
    out = np.empty(muH.shape)
    sbh = np.sqrt(sg ** 2 * hv ** 2 + (2 - sg) ** 2 * B_BODY ** 2 + S_RESID ** 2)[None, :, None, None]
    sa4, sg4 = sa[None, :, None, None], sg[None, :, None, None]
    hb4, hv4 = hb[None, :, None, None], hv[None, :, None, None]
    for i in range(0, muH.shape[0], chunk):
        h = muH[i:i + chunk, :, None, None] + hb4 * _XU[None, None, :, None]
        b = muB[i:i + chunk, :, None, None] + B_BATCH * _XB[None, None, None, :]
        pH = ndtr((h - LN10) / hv4)
        lns = np.minimum(sa4 + sg4 * (h - b - X0), 0.0)
        pB = ndtr((lns + 2.0 * b - LN4 - LN11) / sbh)
        out[i:i + chunk] = (_W[None, None] * (pH * pB) ** 3).sum(axis=(2, 3))
    return out


def margin(post, dr, idx=None):
    """A tie-break score per route and draw: the smaller log margin to the two bars at the median body."""
    muH, muB = post.mu(dr, idx)
    lns = np.minimum(dr["S"][None, :, 0] + dr["S"][None, :, 1] * (muH - muB - X0), 0.0)
    return np.minimum(muH - LN10, lns + 2 * muB - LN4 - LN11)


# ------------------------------------------------------------------------------------------------ requests
def prep_conditions(route):
    if route["preparation"] == "none":
        return {"method": "none", "feedstock_d50_um": float(route["feedstock_d50_um"])}
    return {"method": "vacuum_degassing", "temperature_c": 600, "hold_time_s": 900,
            "feedstock_d50_um": float(route["feedstock_d50_um"])}


def sps_conditions(route, fsid):
    c = {"temperature_c": float(route["temperature_c"]), "heating_rate_c_min": float(route["heating_rate_c_min"]),
         "hold_time_s": float(route["hold_time_s"]), "pressure_mpa": float(route["pressure_mpa"]),
         "pressure_schedule": route["pressure_schedule"], "current_path": route["current_path"],
         "mass_g": economy.FIXED["mass_g"], "diameter_mm": economy.FIXED["diameter_mm"],
         "measurement_context": {"piece_id": PIECE, "density_method": DENSITY_METHOD, "field_set_id": fsid}}
    if route["pressure_full_at_c"] is not None:
        c["pressure_full_at_c"] = float(route["pressure_full_at_c"])
    return c


SCREEN = ["density_g_cm3", "hci_screen_kOe"]
LOOP = ["hci_kOe", "br_kGs", "bhmax_MGOe"]
STAGE_TAG = {"as_sintered": "S", "after_first_anneal": "A", "after_second_anneal": "B"}


def fsid_for(sid, stage, n=1):
    return f"{sid}-{STAGE_TAG[stage]}{n}"


class Planner:
    """Builds requests against a record, numbering new preparations and bodies after those already in it. A new body's
    cycle_id names the declared route it is made for (`<body>-R<route index in economy.declared_routes() order>`),
    so later rounds can anneal it toward that route from the record alone."""

    def __init__(self, arm, rec):
        self.arm, self.rec = arm, rec
        self.n_body = len(rec.own_bodies())
        self.n_prep = sum(1 for p in rec.preps.values() if not p["free"])
        self.new_preps = {}

    def prep_for(self, route, fresh=False):
        """An existing preparation with this route's conditions (REF-P included), or a new one."""
        cond = prep_conditions(route)
        key = (route["preparation"], float(route["feedstock_d50_um"]))
        if not fresh:
            for pid, p in self.rec.preps.items():
                if (p["label"], p["feedstock_d50_um"]) == key:
                    return pid, self._declared(pid, cond)
            if key in self.new_preps:
                return self.new_preps[key], cond
        self.n_prep += 1
        pid = f"{self.arm.upper()}-P{self.n_prep}"
        if not fresh:
            self.new_preps[key] = pid
        return pid, cond

    def _declared(self, pid, cond):
        """A reused preparation is listed again with the conditions of its first declaration."""
        for r in self.rec.returns:
            for p in r["request"].get("preparations", []):
                if p["preparation_id"] == pid:
                    return p["conditions"]
        return cond

    def manufacture(self, rid, routes, rationale, fresh=False, screen=True):
        """New bodies at the routes' manufacture controls (one fresh preparation shared by all if fresh). Measured as
        sintered: density + Hci screen, or density only when screen is False."""
        preps, specs, fresh_pid = {}, [], None
        for i in routes:
            route = ROUTES[i]
            if fresh:
                fresh_pid = fresh_pid or self.prep_for(route, fresh=True)
                pid, cond = fresh_pid
            else:
                pid, cond = self.prep_for(route)
            preps[pid] = cond
            self.n_body += 1
            sid = f"{self.arm.upper()}-B{self.n_body:02d}"
            specs.append({"specimen_id": sid, "preparation_id": pid, "cycle_id": f"{sid}-R{i}",
                          "conditions": sps_conditions(route, fsid_for(sid, "as_sintered")),
                          "measurements": list(SCREEN) if screen else ["density_g_cm3"]})
        req = {"origin": "synthetic", "operation": "manufacture", "lot_id": LOT,
               "preparations": [{"preparation_id": pid, "lot_id": LOT, "conditions": c} for pid, c in preps.items()],
               "specimens": specs}
        return {"request_id": rid, "request": req, "rationale": rationale}


def anneal_request(rec, rid, items, rationale):
    """items: [(sid, step_stage, hold_s, measurements)], all bodies at the same input stage."""
    specs = []
    for sid, st, hold, meas in items:
        T = 900.0 if st == "after_first_anneal" else 520.0
        specs.append({"specimen_id": sid, "preparation_id": rec.bodies[sid]["preparation_id"], "cycle_id": rec.cycle[sid],
                      "conditions": {"anneal_steps": [{"stage": st, "temperature_c": T, "hold_time_s": hold,
                                                       "heating_rate_c_min": 10}],
                                     "mass_g": economy.FIXED["mass_g"],
                                     "measurement_context": {"piece_id": PIECE, "density_method": DENSITY_METHOD,
                                                             "field_set_id": fsid_for(sid, st)}},
                      "measurements": list(meas)})
    req = {"origin": "synthetic", "operation": "anneal", "lot_id": LOT, "input_stage": rec.bodies[items[0][0]]["stage"],
           "output_stage": items[0][1], "specimens": specs}
    return {"request_id": rid, "request": req, "rationale": rationale}


EXTEND_SCREENS = True   # set False for the episode once the laboratory refuses a loop request (see next_action)


def loop_request(rec, rid, sids, rationale):
    """Full loops at the bodies' current stage (one stage per request): the screened acquisition extended to a full loop
    (same field_set_id; charged the difference), else (or after a refused loop request) a new acquisition."""
    stage = rec.bodies[sids[0]]["stage"]
    specs = []
    for sid in sids:
        screened = [a for a in rec.acq_at(sid, stage) if a.hci_kind == "screen"] if EXTEND_SCREENS else []
        fsid = screened[0].fsid if screened else fsid_for(sid, stage, 2)
        specs.append({"specimen_id": sid, "preparation_id": rec.bodies[sid]["preparation_id"], "cycle_id": rec.cycle[sid],
                      "conditions": {"mass_g": economy.FIXED["mass_g"],
                                     "measurement_context": {"piece_id": PIECE, "density_method": DENSITY_METHOD,
                                                             "field_set_id": fsid}},
                      "measurements": list(LOOP) if screened else ["density_g_cm3"] + LOOP})
    req = {"origin": "synthetic", "operation": "measure", "lot_id": LOT, "input_stage": stage, "output_stage": stage,
           "specimens": specs}
    return {"request_id": rid if EXTEND_SCREENS else rid + "n", "request": req, "rationale": rationale}


def by_stage(rec, sids):
    out = {}
    for s in sids:
        out.setdefault(rec.bodies[s]["stage"], []).append(s)
    return sorted(out.items(), key=lambda kv: STAGES.index(kv[0]))


# ------------------------------------------------------------------------------------------------ the arms
class Arm:
    """Round skeleton shared by both arms. rounds() lists (tag, planner, reserve): a planner takes the record of the
    earlier rounds and returns its requests; each request is trimmed (bodies dropped from the end) so the ledger after
    it stays within the allowance less the round's reserve, which holds back credits for the later rounds."""
    name = "x"
    describe = ""

    def __init__(self, state):
        self.state = state
        self.cfg = CONFIG[self.name]
        self.seed = int(state.get("seed", 0))
        self.prior = None

    # ---- model
    def posterior(self, rec):
        return Posterior(self.name, rec, self.prior, self.cfg.get("var_grid"), self.cfg.get("prior_sd_override"))

    def rng(self, tag):
        h = sum((i + 1) * ord(ch) for i, ch in enumerate(tag))
        return np.random.default_rng([self.seed % (2 ** 63), h])

    def candidates(self):
        """Route indices this arm ranks and claims over (bo: every declared route)."""
        return np.arange(R)

    def evaluate(self, rec, tag):
        post = self.posterior(rec)
        dr = post.draws(self.cfg["n_theta"], self.rng("eval-" + tag))
        cand = self.candidates()
        p3 = post.p3(dr, cand)
        full = np.zeros((R, p3.shape[1]))
        full[cand] = p3
        ok = full >= REQUIREMENT["p_all"]
        marg = np.full(R, -9.0)
        marg[cand] = margin(post, dr, cand).mean(axis=1)
        return {"post": post, "meets": ok.mean(axis=1), "mean": full.mean(axis=1), "margin": marg,
                "exists": float(ok[cand].any(axis=0).mean())}

    def rank(self, ev, r):
        """Order of routes: posterior P(meets), then mean p3; with rank_cap c, P(meets) counts only up to c and the
        posterior mean margin to the bars (the smaller log margin of the median body) breaks the tie above it."""
        cap = self.cfg.get("rank_cap")
        if cap is None:
            return (round(float(ev["meets"][r]), 6), round(float(ev["mean"][r]), 6))
        return (round(min(float(ev["meets"][r]), cap), 6), round(float(ev["margin"][r]), 6))

    def measured(self, rec):
        return {a.route for a in rec.acqs if a.route is not None and a.hci and rec.own_acq(a)}

    def candidate(self, rec, ev):
        """The leading route: a verified route if any, else the measured route with the highest P(meets)."""
        ver = verified_routes(rec)
        if ver:
            return max(ver, key=lambda r: (self.rank(ev, r), -r)), "verified"
        meas = self.measured(rec)
        if not meas:
            return None, "none"
        return max(meas, key=lambda r: (self.rank(ev, r), -r)), "measured"

    def explore_routes(self, rec, ev, n, tag):
        """n untested routes for exploration when no candidate is worth replicating (doe: highest posterior mean p3 in
        the design region; bo: Thompson draws)."""
        raise NotImplementedError

    def made_in(self, rec, make_tag):
        out = []
        for r in rec.returns:
            if r["request_id"].startswith(f"{self.name.upper()}-{make_tag}-") \
                    and r["request"].get("operation", "manufacture") == "manufacture":
                out += [s["specimen_id"] for s in r["request"]["specimens"] if s["specimen_id"] in rec.bodies]
        return out

    # ---- shared rounds
    def plan_loops(self, rec, tag, n=None):
        """Full loops on the leading measured states still at their stage (by P(meets), then mean p3), one per route."""
        ev = self.evaluate(rec, tag)
        cands = []
        for sid, b in rec.bodies.items():
            r = rec.current_route(sid)
            acq = rec.acq_at(sid, b["stage"])
            if r is None or any(a.bh is not None for a in acq):
                continue
            if not any(a.hci_kind == "screen" and rec.own_acq(a) for a in acq):
                continue
            cands.append((self.rank(ev, r), sid, r))
        cands.sort(key=lambda t: (-t[0][0], -t[0][1], t[1]))
        picks, seen = [], set()
        for _, sid, r in cands:
            if r not in seen:
                seen.add(r)
                picks.append(sid)
            if len(picks) == (n or self.cfg["loops_leading"]):
                break
        return [loop_request(rec, f"{self.name.upper()}-{tag}-{k}", sids,
                             "Full loops on the leading measured states: Br and (BH)max judge the 11 MGOe limit.")
                for k, (st, sids) in enumerate(by_stage(rec, picks))]

    def at_route(self, rec, r):
        """Own bodies whose current state is route r and whose reading there does not clearly fail: looped and
        passing, or screened (Hci screen >= 9.6 kOe) and not yet looped."""
        out = []
        for sid, b in rec.bodies.items():
            if rec.current_route(sid) != r:
                continue
            acq = [a for a in rec.acq_at(sid, b["stage"]) if rec.own_acq(a)]
            if any(a.bh is not None for a in acq):
                if all(p_pass_reading(a.hci, a.bh) >= PASS_SUPPORT for a in acq if a.bh is not None):
                    out.append(sid)
            elif any(a.hci_kind == "screen" and a.hci >= 9.6 for a in acq):
                out.append(sid)
        return out

    def plan_build(self, rec, tag, need_pass=False):
        """Replicate the candidate route when no route is verified and its P(meets) clears tau_replicate: new bodies on
        a new preparation (two, or one when an own body already sits at the route without a failing reading; the loop
        rounds extend that body's screen). With need_pass, one more body on an existing preparation for a candidate
        with a passing loop and none failing. Otherwise (not need_pass) two bodies at the best untested routes."""
        ev = self.evaluate(rec, tag)
        r, why = self.candidate(rec, ev)
        if why == "verified":
            return []
        pl = Planner(self.name, rec)
        rid = f"{self.name.upper()}-{tag}-0"
        if need_pass:
            # decision rule step 2: a route with a passing loop and none failing gets one more body before the claim
            ok = [q for q, ls in loop_summary(rec).items() if all(p >= PASS_SUPPORT for _, p, _ in ls)]
            if not ok:
                return []
            r = max(ok, key=lambda q: (self.rank(ev, q), -q))
            return [pl.manufacture(rid, [r], f"One more body of {RIDS[r]}: a passing loop, none failing.",
                                   screen=ROUTES[r]["anneal"] == "none")]
        if r is not None and ev["meets"][r] >= self.cfg["tau_replicate"]:
            n = 2 - min(1, len(self.at_route(rec, r)))
            return [pl.manufacture(rid, [r] * n, f"Replicate the candidate {RIDS[r]} (P(meets) {ev['meets'][r]:.2f}) "
                                   "on a new preparation.", fresh=True, screen=ROUTES[r]["anneal"] == "none")]
        picks = self.explore_routes(rec, ev, 2, tag)
        if not picks:
            return []
        return [pl.manufacture(rid, picks, "No candidate worth replicating: the best untested routes.", screen=True)]

    def plan_ladder(self, rec, tag, make_tag, step):
        """Anneal the bodies of `make_tag` one step toward the route they were made for: step 1 the 900 C hold, step 2
        the 520 C step. The route's own stage gets density + Hci screen, an intermediate stage density only."""
        items = []
        for sid in self.made_in(rec, make_tag):
            target = rec.intended.get(sid)
            if target is None or rec.bodies[sid]["stage"] != STAGES[step - 1]:
                continue
            prog = economy.ANNEAL_PROGRAMS[ROUTES[target]["anneal"]]
            if len(prog) < step:
                continue
            st, T, hold = prog[step - 1]
            items.append((sid, st, hold, SCREEN if step == len(prog) else ["density_g_cm3"]))
        return [anneal_request(rec, f"{self.name.upper()}-{tag}-0", items,
                               "Anneal toward the route each body was made for.")] if items else []

    def plan_build_loops(self, rec, tag, make_tag):
        """Loops at the route(s) the bodies of `make_tag` were made for: on those bodies and on any other own body at
        that route with a screen and no loop, unless the screen clearly fails (Hci screen below 9.6 kOe: more than two
        reading sds under 10 kOe)."""
        targets = {rec.intended.get(s) for s in self.made_in(rec, make_tag)} - {None}
        sids = []
        for sid, b in sorted(rec.bodies.items()):
            if rec.current_route(sid) not in targets:
                continue
            acq = [a for a in rec.acq_at(sid, b["stage"]) if rec.own_acq(a)]
            scr = [a.hci for a in acq if a.hci_kind == "screen"]
            if any(a.bh is not None for a in acq) or not scr or scr[0] < 9.6:
                continue
            sids.append(sid)
        return [loop_request(rec, f"{self.name.upper()}-{tag}-{k}", v, "Loops at the replicated route.")
                for k, (st, v) in enumerate(by_stage(rec, sids))]

    def build_rounds(self, prefix, need_pass=False):
        return [(f"{prefix}M", lambda rec: self.plan_build(rec, f"{prefix}M", need_pass), 0),
                (f"{prefix}A", lambda rec: self.plan_ladder(rec, f"{prefix}A", f"{prefix}M", 1), 0),
                (f"{prefix}B", lambda rec: self.plan_ladder(rec, f"{prefix}B", f"{prefix}M", 2), 0),
                (f"{prefix}L", lambda rec: self.plan_build_loops(rec, f"{prefix}L", f"{prefix}M"), 0)]

    # ---- the claim (pre-registered rule)
    def decide(self, rec):
        """1. verified first: a route whose own loops on >= 2 bodies at its own stage all meet both limits (P >= 0.95
        under the 2 % reading model) is nominated (highest posterior P(meets) among verified routes);
        2. (the replication rounds have already spent the budget on a candidate with one passing loop);
        3. restrict when the posterior P(any candidate route meets) <= tau_restrict;
        4. otherwise nominate the measured route with the highest posterior P(meets)."""
        ev = self.evaluate(rec, "final")
        ver = verified_routes(rec)
        if ver:
            r = max(ver, key=lambda r: (self.rank(ev, r), -r))
            return self._nominate(rec, ev, r, f"Own loops on {len({s for s, _, _ in ver[r]})} bodies of this route at "
                                              "its own stage all meet both limits (verified first); P(any declared "
                                              f"route meets) {ev['exists']:.2f}.")
        meas = self.measured(rec)
        r = max(meas, key=lambda r: (self.rank(ev, r), -r)) if meas else None
        if ev["exists"] <= self.cfg["tau_restrict"] or r is None:
            return self._restrict(rec, ev, r)
        return self._nominate(rec, ev, r, f"No route verified; the measured route with the highest posterior P(meets) "
                                          f"({ev['meets'][r]:.2f}); P(any declared route meets) {ev['exists']:.2f} "
                                          f"> {self.cfg['tau_restrict']:.2f}.")

    def _nominate(self, rec, ev, r, why):
        premises = sorted({i for a in rec.acqs if a.route == r and rec.own_acq(a) for i in a.loop_ids})
        h, b, bh = ev["post"].point([r])[0]
        return {"claim": {"form": "nominate", "route": dict(ROUTES[r]), "p_meets": round(float(ev["meets"][r]), 4)},
                "premises": premises,
                "next_action": "Make three bodies of this route from a new preparation and measure a full loop on each; "
                               "a loop below 10 kOe or 11 MGOe would change this claim.",
                "rationale": f"{self.describe}. {why} Posterior-mean median body: Hci {h:.2f} kOe, Br {b:.2f} kGs, "
                             f"(BH)max {bh:.2f} MGOe. The frozen rule was applied unchanged; {rec.ledger['credits']} "
                             f"credits and {rec.ledger['new_bodies']} new bodies used."}

    def _restrict(self, rec, ev, best=None):
        premises = sorted({i for ls in loop_summary(rec).values() for _, _, a in ls for i in a.loop_ids})
        return {"claim": {"form": "restrict", "p_any_route": round(float(ev["exists"]), 4)},
                "premises": premises,
                "next_action": "A full loop meeting 10 kOe and 11 MGOe on any declared route would change this claim; "
                               "the next body would be the highest-ranked untested route.",
                "rationale": f"{self.describe}. No route verified; the policy's posterior P(any declared route meets) is "
                             f"{ev['exists']:.2f} <= {self.cfg['tau_restrict']:.2f}. "
                             + (f"Best measured route {RIDS[best]} (P(meets) {ev['meets'][best]:.2f}). " if best is not None
                                else "") + f"{rec.ledger['credits']} credits spent."}


class DOE(Arm):
    """Design of experiments with a fitted response surface (no supplied model).

    D1  2^(6-3) resolution-III fractional factorial at the fitted preparation, 8 bodies, density + Hci screen:
        A feedstock 4.0/2.0 um (the 4.0 um bodies reuse REF-P), B peak 800/900 C, C load on cooling with the default
        ramp / full at 700 C, D = AB hold 300/900 s, E = AC heating 10/100 C/min, F = BC pressure 60/40 MPa. The
        current path is declared only with load in heating or at the dwell, outside this region, so every body is
        compact. The reference body (850 C) is a ninth point and gives the peak curvature.
    D2  fit the response surface (main effects, t^2, d^2 and the declared interactions) and make the top-ranked
        untested base inside the design region on a preparation already made (steepest ascent; ties by the predicted
        margin), density + screen, within the reserve.
    D3  anneal contrast on the two leading bodies (posterior mean p3 of their as-sintered state, then margin): one
        through 900 C / 1 h + 520 C, the other through 900 C / 4 h + 520 C, density + screen at every stage (the
        reference body joins when the budget carries it through both steps).
    D4  full loops on the two leading measured states (their screens extended).
    R   replicate the candidate on a new preparation (two bodies, or one when an own body already sits at the route
        without a failing reading; annealed to its program), loops at the route.
    X, Y  one more body for the leading route with a passing loop and none failing (decision rule step 2), twice at
          most, if the budget allows. Then the claim.
    """
    name = "doe"
    describe = "Design of experiments (fractional-factorial screen, fitted response surface, anneal contrast, replication)"

    def candidates(self):
        """The design region: fitted preparation, peak 800-900 C, load on cooling (either ramp, compact current path);
        the other controls were design factors (feedstock 3.0 um interpolates)."""
        return DOE_REGION

    def rounds(self):
        c = self.cfg
        return ([("D1", self.plan_d1, c["reserve_d1"]), ("D2", self.plan_d2, c["reserve_d1"]),
                 ("D3A", lambda rec: self.plan_contrast(rec, "D3A", 1), c["reserve_d3"]),
                 ("D3B", lambda rec: self.plan_contrast(rec, "D3B", 2), c["reserve_d3"]),
                 ("D4", lambda rec: self.plan_loops(rec, "D4"), c["reserve_build"])]
                + self.build_rounds("R") + self.build_rounds("X", need_pass=True) + self.build_rounds("Y", need_pass=True))

    def plan_d1(self, rec):
        routes = []
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    d, e, f = a * b, a * c, b * c
                    route = {"preparation": "fitted", "feedstock_d50_um": 4.0 if a < 0 else 2.0,
                             "temperature_c": 800.0 if b < 0 else 900.0, "hold_time_s": 300.0 if d < 0 else 900.0,
                             "heating_rate_c_min": 10.0 if e < 0 else 100.0, "pressure_mpa": 60.0 if f < 0 else 40.0,
                             "pressure_schedule": "on_cooling", "pressure_full_at_c": None if c < 0 else 700.0,
                             "current_path": "compact", "anneal": "none"}
                    routes.append(INDEX[economy.route_id(route)])
        return [Planner(self.name, rec).manufacture("DOE-D1-0", routes, "2^(6-3) screening design.")]

    def surface_order(self, rec, ev, as_sintered=True, existing_preps=False):
        """Untested in-region routes by posterior mean p3, then mean margin; one per base."""
        built = {BASE_OF[a.route] for a in rec.acqs if a.route is not None}
        have = {(p["label"], p["feedstock_d50_um"]) for p in rec.preps.values()}
        order = sorted(self.candidates(), key=lambda i: (-round(float(ev["mean"][i]), 4),
                                                         -round(float(ev["margin"][i]), 6), i))
        out, bases = [], set()
        for i in order:
            if (as_sintered and ROUTES[i]["anneal"] != "none") or BASE_OF[i] in built or BASE_OF[i] in bases:
                continue
            if existing_preps and (ROUTES[i]["preparation"], ROUTES[i]["feedstock_d50_um"]) not in have:
                continue
            bases.add(BASE_OF[i])
            out.append(int(i))
        return out

    def plan_d2(self, rec):
        ev = self.evaluate(rec, "D2")
        picks = self.surface_order(rec, ev, existing_preps=True)[:self.cfg["steepest_ascent_bodies"]]
        return [Planner(self.name, rec).manufacture("DOE-D2-0", picks, "Steepest ascent on the fitted surface.")]

    def explore_routes(self, rec, ev, n, tag):
        return self.surface_order(rec, ev, as_sintered=False)[:n]

    def plan_contrast(self, rec, tag, step):
        """Anneal contrast on the leading bodies (highest posterior mean p3 of their as-sintered state; the reference
        body REF-B1 last), each through the configured program (900 C hold, then 520 C) with density + screen at every
        stage, as many as the budget carries through both steps above the round's reserve (20 credits per body and
        step). Step 2 follows on every body that took step 1."""
        if step == 1:
            ev = self.evaluate(rec, tag)
            sids = [s for s in rec.own_bodies() if rec.bodies[s]["stage"] == "as_sintered"
                    and rec.current_route(s) is not None]
            sids.sort(key=lambda s: (-round(float(ev["mean"][rec.current_route(s)]), 4),
                                     -round(float(ev["margin"][rec.current_route(s)]), 6), s))
            sids = sids[:self.cfg["contrast_bodies"]] + ["REF-B1"]
            per = 2 * (economy.PRICES["anneal_step"] + economy.PRICES["density"] + economy.PRICES["hci_screen"])
            k = max(0, (economy.ALLOWANCE["credits"] - self.cfg["reserve_d3"] - rec.ledger["credits"]) // per)
            sids = [s for s in sids if s in rec.bodies and rec.bodies[s]["stage"] == "as_sintered"][:k]
            progs = [self.cfg["anneal_contrast"][i % len(self.cfg["anneal_contrast"])] for i in range(len(sids))]
        else:
            sids, progs = [], []
            for r in rec.returns:
                if r["request_id"].startswith("DOE-D3A-"):
                    for s in r["request"]["specimens"]:
                        if rec.bodies[s["specimen_id"]]["stage"] == "after_first_anneal":
                            sids.append(s["specimen_id"])
                            progs.append("900_1h+520")
        items = []
        for sid, prog in zip(sids, progs):
            st, T, hold = economy.ANNEAL_PROGRAMS[prog][step - 1]
            items.append((sid, st, hold, SCREEN))
        return [anneal_request(rec, f"DOE-{tag}-0", items, "Within-body anneal contrast on the leading bodies.")] \
            if items else []


class BO(Arm):
    """Bayesian optimisation: Thompson sampling on P(3 of 3 fresh-preparation bodies meet both limits) under the shared
    model, with the supplied C7.3 model's per-route medians (both magnetic calculations, weighted by their marginal
    likelihood on the record) as the prior mean; the surrogate learns the discrepancy from the policy's own readings.

    E1, E2  Thompson rounds of new bodies (4, then 3 within the reserve), each a full declared route under its own
            posterior draw (base and anneal program), made with density + Hci screen. A fresh draw then decides whether
            to stop as sintered or anneal toward the drawn hold (900 C only: density + screen; with 520 C: density at
            900 C, then the 520 C step with density + screen). In E1 the reference body REF-B1 joins the anneal step.
    L       full loops on the two leading measured states (their screens extended).
    R       replicate the candidate on a new preparation (two bodies, annealed to its program), loops.
    X, Y    one more body for the leading route with a passing loop and none failing (decision rule step 2), twice at
            most, if the budget allows. Then the claim.
    """
    name = "bo"
    describe = ("Bayesian optimisation (Thompson sampling on P(3 of 3) with the supplied C7.3 model as prior mean and a "
                "learned discrepancy)")

    def __init__(self, state):
        super().__init__(state)
        self.prior, self.prior_found = load_prior(state.get("prior"))

    def rounds(self):
        c = self.cfg
        out = []
        if c.get("start_design"):
            out += [("E0M", self.plan_start, c["reserve_explore"]),
                    ("E0A", lambda rec: self.plan_anneal(rec, "E0A", "E0M", True, any_program=True), c["reserve_explore"]),
                    ("E0B", lambda rec: self.plan_second(rec, "E0B", "E0M"), c["reserve_explore"])]
        for k, n in enumerate(c["explore_rounds"]):
            t = f"E{k + 1}"
            ref = k == 0 and not c.get("start_design")
            out += [(f"{t}M", lambda rec, t=t, n=n: self.plan_explore(rec, t + "M", n), c["reserve_explore"]),
                    (f"{t}A", lambda rec, t=t, ref=ref: self.plan_anneal(rec, t + "A", t + "M", ref), c["reserve_explore"]),
                    (f"{t}B", lambda rec, t=t: self.plan_second(rec, t + "B", t + "M"), c["reserve_explore"])]
        return (out + [("L", lambda rec: self.plan_loops(rec, "L"), c["reserve_build"])]
                + self.build_rounds("R") + self.build_rounds("X", need_pass=True) + self.build_rounds("Y", need_pass=True))

    def thompson(self, rec, post, n, rng, exclude=()):
        """n routes, each the best untested base (with its best anneal program) under its own posterior draw."""
        built = {BASE_OF[a.route] for a in rec.acqs if a.route is not None} | set(exclude)
        picks = []
        for _ in range(n):
            dr = post.draws(1, rng)
            score = post.p3(dr)[:, 0] + 1e-3 * np.tanh(margin(post, dr)[:, 0])
            for i in np.argsort(-score, kind="stable"):
                if BASE_OF[i] not in built:
                    built.add(BASE_OF[i])
                    picks.append(int(i))
                    break
        return picks

    def plan_explore(self, rec, tag, n):
        """n Thompson picks, kept in draw order while each body can be carried through its drawn program (body,
        density and screen as sintered, the anneal steps with their readings, a new preparation where needed) above the
        exploration reserve."""
        picks = self.thompson(rec, self.posterior(rec), n, self.rng("ts-" + tag))
        P = economy.PRICES
        budget = economy.ALLOWANCE["credits"] - self.cfg["reserve_explore"] - rec.ledger["credits"]
        have = {(p["label"], p["feedstock_d50_um"]) for p in rec.preps.values()}
        kept = []
        for i in picks:
            steps = len(economy.ANNEAL_PROGRAMS[ROUTES[i]["anneal"]])
            key = (ROUTES[i]["preparation"], ROUTES[i]["feedstock_d50_um"])
            cost = (P["body"] + P["density"] + P["hci_screen"] + (P["preparation"] if key not in have else 0)
                    + steps * (P["anneal_step"] + P["density"]) + (P["hci_screen"] if steps else 0))
            if cost > budget:
                continue
            budget -= cost
            have.add(key)
            kept.append(i)
        if not kept:
            return []
        return [Planner(self.name, rec).manufacture(f"BO-{tag}-0", kept, "Thompson-sampling picks (one posterior "
                                                    "draw per body): " + "; ".join(RIDS[i] for i in kept))]

    def explore_routes(self, rec, ev, n, tag):
        return self.thompson(rec, ev["post"], n, self.rng("ts-" + tag))

    def plan_start(self, rec):
        """Space-filling start (A2's design round, here a 2^(3-1) half fraction on the three levers with the widest
        declared spread): feedstock 4.0/2.0 um x peak 800/900 C x load on cooling default / full at 700 C, the other
        controls at the reference body's levels; the anneal program of each body is then chosen by a Thompson draw."""
        routes = []
        for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            c = a * b
            route = {"preparation": "fitted", "feedstock_d50_um": 4.0 if a < 0 else 2.0,
                     "temperature_c": 800.0 if b < 0 else 900.0, "hold_time_s": 300.0, "heating_rate_c_min": 10.0,
                     "pressure_mpa": 60.0, "pressure_schedule": "on_cooling",
                     "pressure_full_at_c": None if c > 0 else 700.0, "current_path": "compact", "anneal": "none"}
            routes.append(INDEX[economy.route_id(route)])
        return [Planner(self.name, rec).manufacture("BO-E0M-0", routes, "Space-filling start (2^(3-1)).")]

    def plan_anneal(self, rec, tag, make_tag, with_reference, any_program=False):
        """Stop as sintered or anneal toward a 900 C program, by a fresh Thompson draw per body over its as-sintered
        route and the anneal programs reachable from it (the drawn hold for a body made for an annealed route; every
        program for the reference body)."""
        post = self.posterior(rec)
        rng = self.rng("ann-" + tag)
        sids = self.made_in(rec, make_tag) + (["REF-B1"] if with_reference and "REF-B1" in rec.bodies else [])
        items = []
        budget = economy.ALLOWANCE["credits"] - self.cfg["reserve_explore"] - rec.ledger["credits"]
        step = economy.PRICES["anneal_step"] + economy.PRICES["density"]
        for sid in sids:
            cur = rec.current_route(sid)
            if cur is None or rec.bodies[sid]["stage"] != "as_sintered":
                continue
            target = None if any_program else rec.intended.get(sid)
            if target is not None:
                if ROUTES[target]["anneal"] == "none":
                    continue
                h = ROUTES[target]["anneal"].split("+")[0]
                progs = [h, h + "+520"]
            else:
                progs = [a for a in ANNEALS if a != "none"]
            idx = np.array([cur] + [with_anneal(ROUTES[cur], a) for a in progs])
            dr = post.draws(1, rng)
            sc = post.p3(dr, idx)[:, 0] + 1e-3 * np.tanh(margin(post, dr, idx)[:, 0])
            best = int(np.argmax(sc[1:]))
            if sc[0] > sc[1 + best] + self.cfg["anneal_margin"]:
                continue   # the fresh draw clearly prefers stopping as sintered; otherwise follow the drawn program
            prog = progs[best]
            st, T, hold = economy.ANNEAL_PROGRAMS[prog][0]
            # this step, plus the committed 520 C step with its screen: kept only if both fit above the reserve
            cost = step + (step + economy.PRICES["hci_screen"] if "+" in prog else economy.PRICES["hci_screen"])
            if cost > budget:
                continue
            budget -= cost
            items.append((sid, st, hold, SCREEN if "+" not in prog else ["density_g_cm3"]))
        return [anneal_request(rec, f"BO-{tag}-0", items, "Anneal toward the program a fresh Thompson draw prefers.")] \
            if items else []

    def plan_second(self, rec, tag, make_tag):
        """The 520 C step for the bodies whose 900 C stage was read for density only (committed to the 520 C step)."""
        items = []
        for sid in self.made_in(rec, make_tag) + ["REF-B1"]:
            b = rec.bodies.get(sid)
            if b is None or b["stage"] != "after_first_anneal" or rec.current_route(sid) is None:
                continue
            if any(a.hci for a in rec.acq_at(sid, "after_first_anneal")) or not rec.acq_at(sid, "after_first_anneal"):
                continue
            items.append((sid, "after_second_anneal", 7200.0, SCREEN))
        return [anneal_request(rec, f"BO-{tag}-0", items, "The 520 C step.")] if items else []


ARMS = {"doe": DOE, "bo": BO}
DOE_REGION = np.array([i for i, r in enumerate(ROUTES) if r["preparation"] == "fitted" and r["temperature_c"] >= 800.0
                       and r["pressure_schedule"] == "on_cooling"])


# ------------------------------------------------------------------------------------------------ protocol
def history_request_ids(history):
    """request_ids of every request_simulation the runner attempted, in order, whatever its status. Accepts history rows
    {"action": {"tool", "arguments"}, ...} or flat {"tool", "arguments"} / {"request_id"} rows."""
    out = []
    for h in history or []:
        act = h.get("action") if isinstance(h.get("action"), dict) else h
        if (act or {}).get("tool") not in (None, "request_simulation"):
            continue
        rid = ((act or {}).get("arguments") or {}).get("request_id") or h.get("request_id")
        if rid:
            out.append(rid)
    return out


def trim(rec, req, contract, reserve=0):
    """Drop bodies from the end of a request until the ledger after it is within the allowance less `reserve`
    (economy.ledger over the returns plus this request: the contract's own arithmetic)."""
    allowance = (contract or {}).get("allowance") or economy.ALLOWANCE
    req = json.loads(json.dumps(req))
    body = req["request"]
    while body["specimens"]:
        used = {s["preparation_id"] for s in body["specimens"]}
        if "preparations" in body:
            body["preparations"] = [p for p in body["preparations"] if p["preparation_id"] in used]
        led = economy.ledger(rec.entries + [{"request_id": req["request_id"], "request": body, "delivered": True}])
        if led["credits"] <= allowance["credits"] - reserve and led["new_bodies"] <= allowance["bodies"]:
            return req
        body["specimens"].pop()
    return None


def apply_dev_config(state, arm):
    """Development variants only: `dev_config` is absent from every production
    STATE.json, so the frozen configuration is CONFIG and the module constants above."""
    dev = state.get("dev_config") or {}
    arm.cfg.update(dev.get("cfg") or {})
    for k, v in (dev.get("model") or {}).items():
        if k in ("H_BATCH", "H_BODY", "B_BATCH", "B_BODY", "SCREEN_SD", "S_RESID", "KAPPA_SD"):
            globals()[k] = float(v)


def refused_measures(state):
    """Loop requests (operation measure) the laboratory refused earlier in this episode."""
    out = 0
    for h in state.get("history") or []:
        act = h.get("action") if isinstance(h.get("action"), dict) else h
        res = h.get("result") if isinstance(h.get("result"), dict) else h
        if (act or {}).get("tool") == "request_simulation" and res.get("status") == "refused" \
                and ((act.get("arguments") or {}).get("request") or {}).get("operation") == "measure":
            out += 1
    return out


def next_action(state):
    global EXTEND_SCREENS
    arm = ARMS[state["arm"]](state)
    apply_dev_config(state, arm)
    # If extending a screened acquisition to a loop is ever refused, measure loops as new acquisitions (50 credits).
    EXTEND_SCREENS = refused_measures(state) == 0
    full = Record(state.get("returns") or [])
    attempted = set(history_request_ids(state.get("history")))
    rounds = arm.rounds()
    tags = [t for t, _, _ in rounds]
    prefix = arm.name.upper() + "-"
    started = {t for t in tags for rid in attempted if rid.startswith(f"{prefix}{t}-")}
    for k, (tag, planner, reserve) in enumerate(rounds):
        if any(t in started for t in tags[k + 1:]):
            continue   # a later round has begun: this one is complete
        earlier = tuple(f"{prefix}{t}-" for t in tags[:k])
        rec = Record([r for r in full.returns if r["request_id"] in economy.FREE_REQUESTS
                      or r["request_id"].startswith(earlier)])
        for req in planner(rec):
            if req["request_id"] in attempted:
                continue
            body = req["request"]
            if body["operation"] != "manufacture":   # bodies refused earlier, or no longer at the input stage, drop out
                body["specimens"] = [s for s in body["specimens"] if s["specimen_id"] in full.bodies
                                     and full.bodies[s["specimen_id"]]["stage"] == body["input_stage"]]
            req = trim(full, req, state.get("contract"), reserve)
            if req is None:
                continue
            return {"tool": "request_simulation",
                    "arguments": {"request_id": req["request_id"], "simulation": "magnetic",
                                  "rationale": req["rationale"], "request": req["request"]}}
    return {"tool": "decide", "decision": arm.decide(full)}


def main():
    with open(sys.argv[1]) as f:
        state = json.load(f)
    sys.stdout.write(json.dumps(next_action(state)) + "\n")


if __name__ == "__main__":
    main()
