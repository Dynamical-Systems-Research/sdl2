"""SDL-2 v1.2 economy and coverage from an agent's own executed requests. Public: uses only the request echo and the
public delivery each request returned. Shared by the episode's COVERAGE.md and by the grader (overspend, and which
measured states are declared routes).

Prices, the allowance and the declared routes are CONTRACT.md's. A preparation is charged once, when it first appears. A density reading
or loop is charged once per acquisition (specimen, piece, stage, field_set_id; density also by method), matching the
world's reading identity: a new piece or a new field_set_id is a new, charged reading. Extending an Hci-screen
acquisition to a full loop is charged the difference. SEM and polished-section characterization is charged
once per specimen and request. sps_log, the process record of every manufacture, is free. The reference body REF-B1
(request REF-INITIAL) is free and is not one of the 12 new bodies; its preparation REF-P may be reused.
"""
from __future__ import annotations

PRICES = {"preparation": 15, "body": 10, "anneal_step": 6, "density": 2, "loop": 50, "hci_screen": 12,
          "loop_upgrade": 38, "characterization": 8}
ALLOWANCE = {"credits": 570, "bodies": 12}
LOOP_QUANTITIES = {"hci_kOe", "br_kGs", "bhmax_MGOe", "demagnetization_curve"}
CHARACTERIZATION = {"grain_size_nm", "grain_d90_nm", "coarse_number_fraction", "porosity_area_fraction", "microstructure",
                    "boundary_chemistry"}
FREE_REQUESTS = {"REF-INITIAL"}
# Declared routes: one value of every control the C7.3 world responds to (routes/ROUTE_EXPANSION.md). Keys are the
# request's own control names; `anneal` names an executed anneal program. pressure_full_at_c (the temperature at which
# full load is reached while cooling) is null except in the slow on-cooling program. The current path is declared both
# ways only where load is applied in heating or at the dwell: with load on cooling the package's current-dose term is
# zero, so an insulated twin would be the same body (routes/ROUTE_EXPANSION.md). Charge mass and die diameter are
# fixed: C7.3's density and magnetic calculations do not read them (routes/SCAN_SUMMARY.json, "inert").
ROUTE_KEYS = ("preparation", "feedstock_d50_um", "temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa",
              "pressure_schedule", "pressure_full_at_c", "current_path", "anneal")
DECLARED = {"preparation": ("none", "fitted"), "feedstock_d50_um": (4.0, 3.0, 2.0),
            "temperature_c": (750.0, 800.0, 850.0, 900.0), "hold_time_s": (300.0, 900.0),
            "heating_rate_c_min": (10.0, 100.0), "pressure_mpa": (40.0, 60.0),
            "pressure_schedule": ("on_heating", "on_dwell", "on_cooling"), "pressure_full_at_c": (None, 700.0),
            "current_path": ("compact", "insulated"),
            "anneal": ("none", "900_1h", "900_2h", "900_4h", "900_1h+520", "900_2h+520", "900_4h+520")}
FIXED = {"mass_g": 0.9, "diameter_mm": 10.0}
FIRST, SECOND = "after_first_anneal", "after_second_anneal"
ANNEAL_PROGRAMS = {"none": [],
                   **{f"900_{h}h": [(FIRST, 900.0, 3600.0 * h)] for h in (1, 2, 4)},
                   **{f"900_{h}h+520": [(FIRST, 900.0, 3600.0 * h), (SECOND, 520.0, 7200.0)] for h in (1, 2, 4)}}
STAGE_OF_STEPS = ("as_sintered", "after_first_anneal", "after_second_anneal")


def admissible(route):
    """The declared combinations: a load-reached temperature only with load on cooling; the insulated path only with
    load in heating or at the dwell."""
    if route["pressure_full_at_c"] is not None and route["pressure_schedule"] != "on_cooling":
        return False
    return not (route["current_path"] == "insulated" and route["pressure_schedule"] == "on_cooling")


def declared_routes():
    """Every declared route, in a fixed order (the declared order of each control; anneal varies fastest)."""
    import itertools
    return [r for r in (dict(zip(ROUTE_KEYS, values)) for values in itertools.product(*(DECLARED[k] for k in ROUTE_KEYS)))
            if admissible(r)]


def route_id(route):
    return "|".join(str(route[k]) for k in ROUTE_KEYS)


def anneal_program(steps):
    """The declared anneal program an executed step sequence is, or None."""
    steps = [(s, _num(t), _num(h)) for s, t, h in steps]
    for name, program in ANNEAL_PROGRAMS.items():
        if steps == program:
            return name
    return None


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _prep_label(conditions):
    if conditions.get("method") == "none":
        return "none"
    if conditions.get("method") == "vacuum_degassing" and conditions.get("temperature_c") == 600 \
            and conditions.get("hold_time_s") == 900:
        return "fitted"
    return "other"


def ledger(entries):
    """entries: [{"request_id", "request", "delivered": bool}] in order. Returns spend, bodies and per-body histories."""
    seen, spend = set(), {k: 0 for k in PRICES}
    preps, bodies, acquisitions = {}, {}, set()
    for e in entries:
        rid, req = e["request_id"], e["request"]
        if not e.get("delivered") or rid in seen:
            continue
        seen.add(rid)
        free = rid in FREE_REQUESTS
        op = req.get("operation", "manufacture")
        for p in req.get("preparations", []) if op == "manufacture" else []:
            if p["preparation_id"] in preps:   # a reused preparation is listed again with identical conditions
                continue
            c = p.get("conditions", {})
            preps[p["preparation_id"]] = {"label": _prep_label(c), "feedstock_d50_um": _num(c.get("feedstock_d50_um", 4.0)),
                                          "free": free}
            spend["preparation"] += 0 if free else 1
        for s in req.get("specimens", []):
            sid, c = s["specimen_id"], s.get("conditions", {})
            if op == "manufacture":
                bodies[sid] = {"preparation_id": s["preparation_id"], "free": free, "stage": "as_sintered",
                               "stages": ["as_sintered"], "steps": [],
                               "controls": {k: _num(c.get(k)) for k in ("temperature_c", "hold_time_s", "pressure_schedule",
                                                                        "pressure_mpa", "heating_rate_c_min", "mass_g",
                                                                        "diameter_mm", "current_path", "pressure_full_at_c")},
                               "measured": []}
                spend["body"] += 0 if free else 1
            body = bodies.get(sid)
            if op == "anneal":
                steps = c.get("anneal_steps", [])
                spend["anneal_step"] += 0 if free else len(steps)
                if body:
                    body["steps"] += [(st.get("stage"), _num(st.get("temperature_c")), _num(st.get("hold_time_s")))
                                      for st in steps]
                    body["stage"] = req.get("output_stage", body["stage"])
                    body["stages"].append(body["stage"])
            stage = body["stage"] if body else req.get("output_stage", "as_sintered")
            wanted = set(s.get("measurements", req.get("measurements", [])))
            ctx = c.get("measurement_context", {})
            acq, piece = ctx.get("field_set_id"), ctx.get("piece_id", "1")
            if "density_g_cm3" in wanted:
                key = ("density", sid, piece, stage, acq, ctx.get("density_method", "archimedes"))
                if key not in acquisitions:
                    acquisitions.add(key)
                    spend["density"] += 0 if free else 1
            if wanted & LOOP_QUANTITIES:
                key = ("loop", sid, piece, stage, acq)
                if key not in acquisitions:
                    acquisitions.add(key)
                    if not free and ("screen", sid, piece, stage, acq) in acquisitions:
                        spend["loop_upgrade"] += 1   # a screened acquisition extended to a full loop
                    else:
                        spend["loop"] += 0 if free else 1
            elif "hci_screen_kOe" in wanted:
                key = ("screen", sid, piece, stage, acq)
                if key not in acquisitions and ("loop", sid, piece, stage, acq) not in acquisitions:
                    acquisitions.add(key)
                    spend["hci_screen"] += 0 if free else 1
            if wanted & CHARACTERIZATION:
                spend["characterization"] += 0 if free else 1
            if body and wanted:
                body["measured"].append({"stage": stage, "request_id": rid, "piece_id": ctx.get("piece_id"),
                                         "acquisition": acq, "quantities": sorted(wanted)})
    credits = sum(PRICES[k] * n for k, n in spend.items())
    new_bodies = sum(1 for b in bodies.values() if not b["free"])
    return {"credits": credits, "new_bodies": new_bodies, "counts": spend, "preparations": preps, "bodies": bodies,
            "within_allowance": credits <= ALLOWANCE["credits"] and new_bodies <= ALLOWANCE["bodies"]}


def declared_route(body, preps):
    """The declared route a body's current state is, or None if any executed control is off the declared set-points
    (an undeclared anneal program, charge mass or die diameter included; an omitted current path is the instrument's
    compact default and an omitted diameter its 10 mm default)."""
    p = preps.get(body["preparation_id"], {})
    c = body["controls"]
    values = {"preparation": p.get("label"), "feedstock_d50_um": p.get("feedstock_d50_um"),
              **{k: c.get(k) for k in ("temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa",
                                       "pressure_schedule", "pressure_full_at_c")},
              "current_path": c.get("current_path") or "compact", "anneal": anneal_program(body.get("steps", []))}
    for k in ROUTE_KEYS:
        values[k] = _num(values[k])
        if values[k] not in DECLARED[k]:
            return None
    if not admissible(values):
        return None
    if body["stage"] != STAGE_OF_STEPS[len(ANNEAL_PROGRAMS[values["anneal"]])]:
        return None
    if any((v if k == "diameter_mm" and c.get(k) is None else c.get(k)) != v for k, v in FIXED.items()):
        return None
    return values


def coverage_markdown(state):
    """COVERAGE.md: spend, bodies and the declared controls your own measured states cover. No hidden information."""
    lines = ["# Coverage (refreshed after every request)", "",
             f"Spent {state['credits']} of {ALLOWANCE['credits']} credits; {state['new_bodies']} of {ALLOWANCE['bodies']} new bodies. "
             + ("Within the allowance." if state["within_allowance"] else "Over the allowance: a final claim would fail."), "",
             "| Item | Count | Credits |", "|---|---|---|"]
    for k, n in state["counts"].items():
        lines.append(f"| {k.replace('_', ' ')} | {n} | {n * PRICES[k]} |")
    lines += ["", "## Bodies", "", "| Body | Preparation | SPS controls | Current stage | Measured (stage: quantities) |",
              "|---|---|---|---|---|"]
    looped = {}
    for sid, b in state["bodies"].items():
        p = state["preparations"].get(b["preparation_id"], {})
        c = b["controls"]
        load = c.get("pressure_schedule") + (f" (full at {c['pressure_full_at_c']} °C)" if c.get("pressure_full_at_c") else "")
        ctrl = (f"{c.get('temperature_c')} °C, {c.get('hold_time_s')} s, {c.get('heating_rate_c_min')} °C/min, "
                f"{c.get('pressure_mpa')} MPa, {load}, {c.get('current_path') or 'compact'}")
        meas = "; ".join(f"{m['stage']}: {', '.join(m['quantities'])}" for m in b["measured"]) or "nothing yet"
        lines.append(f"| {sid}{' (reference)' if b['free'] else ''} | {b['preparation_id']} ({p.get('label')}, "
                     f"{p.get('feedstock_d50_um')} µm) | {ctrl} | {b['stage']} | {meas} |")
    for sid, b in state["bodies"].items():
        for m in b["measured"]:
            # loops that can judge 11 MGOe, from the agent's own requests (a paid loop on the reference body counts, as
            # in the grader; the free reference reading does not)
            if {"bhmax_MGOe", "demagnetization_curve"} & set(m["quantities"]) and m["request_id"] not in FREE_REQUESTS:
                steps = b["steps"][:STAGE_OF_STEPS.index(m["stage"])]
                r = declared_route({**b, "stage": m["stage"], "steps": steps}, state["preparations"])
                if r:
                    looped.setdefault(route_id(r), set()).add(sid)
    lines += ["", "## Declared routes with at least one full loop (with (BH)max) from your own requests, at the route's own stage", ""]
    if looped:
        lines += ["| Route (" + ", ".join(ROUTE_KEYS) + ") | Bodies with a loop |", "|---|---|"]
        lines += [f"| {rid.replace('|', ', ')} | {len(s)} ({', '.join(sorted(s))}) |" for rid, s in sorted(looped.items())]
    else:
        lines.append("None yet.")
    lines += ["", "## Declared values not yet built in any body", ""]
    built = {k: set() for k in ROUTE_KEYS}
    for b in state["bodies"].values():
        p = state["preparations"].get(b["preparation_id"], {})
        c = b["controls"]
        for k, v in (("preparation", p.get("label")), ("feedstock_d50_um", p.get("feedstock_d50_um")),
                     ("current_path", c.get("current_path") or "compact"),
                     *((k, c.get(k)) for k in ("temperature_c", "hold_time_s", "heating_rate_c_min", "pressure_mpa",
                                               "pressure_schedule", "pressure_full_at_c"))):
            built[k].add(_num(v))
        built["anneal"] |= {anneal_program(b["steps"][:n]) for n in range(len(b["steps"]) + 1)}
    for k in ROUTE_KEYS:
        missing = [v for v in DECLARED[k] if v not in built[k]]
        lines.append(f"- `{k}`: {', '.join(map(str, missing)) if missing else 'all built'}")
    lines += ["", "This table counts what you built and measured. It says nothing about which routes meet the requirement."]
    return "\n".join(lines) + "\n"
