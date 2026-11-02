# SDL-2 investigation

Dynamical's program objective is reproducible Dy/Tb-free NdFeB with Hci ≥ 20 kOe and (BH)max ≥ 45 MGOe together. This laboratory makes isotropic spark-plasma-sintered bodies, which cannot reach 45 MGOe, so this investigation decides an intermediate requirement for one virtual powder lot.

## The decision

Find out whether any declared route makes magnets from this lot that meet the requirement, and finish with one claim.

**Requirement.** Bodies measured at 20 °C with Hci ≥ 10 kOe and (BH)max ≥ 11 MGOe. A route meets the requirement if three bodies made by it from a new preparation of this lot all meet both limits with probability at least 0.95. The owner checks a nominated route on a confirmation lot you do not see.

**Declared routes.** A route is one value of each control below; there are 8,064. Fixed for every route: a 0.9 g charge in the 10 mm die. The control names are the request's own.

| Control | Declared values |
|---|---|
| `preparation` | `none` (no degassing) or `fitted` (vacuum degassing at 600 °C for 900 s) |
| `feedstock_d50_um` | 4.0, 3.0 or 2.0 |
| `temperature_c` (SPS peak) | 750, 800, 850, 900 |
| `hold_time_s` (at peak) | 300 or 900 |
| `heating_rate_c_min` | 10 or 100 |
| `pressure_mpa` | 40 or 60 |
| `pressure_schedule` and `pressure_full_at_c` | `on_heating`; `on_dwell`; `on_cooling` with `pressure_full_at_c` null (the instrument's default 0.3 min load ramp); `on_cooling` with `pressure_full_at_c` 700 (full load reached at 700 °C while cooling) |
| `current_path` | `compact` (the loaded compact carries the current), or `insulated` with `on_heating` or `on_dwell` |
| `anneal` | `none`; `900_1h`, `900_2h`, `900_4h` (900 °C for 1, 2 or 4 h); `900_1h+520`, `900_2h+520`, `900_4h+520` (then 520 °C for 2 h) |

**Claims.** Write `decision.json` in your workspace, then call `record_decision` with `stop`; the investigation ends at that call (`wait` also ends it here: nothing external is pending).
- **Nominate** one declared route you expect to meet the requirement.
- **Restrict**: claim that no declared route meets it. A restriction covers all 8,064 routes and is judged against the lot's truth, as a nomination is.
- **none** is allowed but is not a claim: it delivers neither a route nor a restriction and never meets the requirement.

About half of the lots in this program have at least one declared route that meets the requirement. Each lot is drawn from the posterior the supplied C7.3 package samples, never one of its stored rows, and selected for this program; in the package's own posterior, passing lots are rarer.

```json
{"claim": {"form": "nominate", "route": {"preparation": "<none|fitted>", "feedstock_d50_um": <4.0|3.0|2.0>,
           "temperature_c": <750|800|850|900>, "hold_time_s": <300|900>, "heating_rate_c_min": <10|100>,
           "pressure_mpa": <40|60>, "pressure_schedule": "<on_heating|on_dwell|on_cooling>",
           "pressure_full_at_c": <null|700>, "current_path": "<compact|insulated>",
           "anneal": "<none|900_1h|900_2h|900_4h|900_1h+520|900_2h+520|900_4h+520>"},
           "p_meets": <your probability>},
 "premises": ["<observation_id>", "..."],
 "next_action": "the next experiment or measurement you would run, and why",
 "rationale": "what you concluded, whether you revised or retained your earlier plan, which returns changed it, and which return would change this claim"}
```
For a restriction use `{"form": "restrict", "p_any_route": <your probability that some route meets>}`; for none, `{"form": "none"}`. `premises` lists the observation IDs your claim rests on.

## Resources

You have 570 credits and 12 new bodies. Spending beyond either is not blocked, but the final claim then fails.

| Item | Credits |
|---|---|
| New preparation (powder batch) | 15 |
| New body | 10 |
| Anneal step, per body | 6 |
| Density reading | 2 |
| Full loop (`hci_kOe`, `br_kGs`, `bhmax_MGOe`, `demagnetization_curve` from one acquisition) | 50 |
| Hci screen (`hci_screen_kOe`) | 12 |
| Extending a screened acquisition to a full loop | 38 |
| SEM or polished-section characterization (`grain_size_nm`, `grain_d90_nm`, `coarse_number_fraction`, `porosity_area_fraction`, `microstructure`, `boundary_chemistry`), per body and request | 8 |

`sps_log`, the process record of a manufacture, is free. A density reading or loop is charged once per acquisition: one specimen, piece, stage and `field_set_id`. Repeating an unchanged request redelivers it. Reusing the same specimen, piece, stage and `field_set_id` in a different request is refused, because the ledger already holds that reading; a new piece or a new `field_set_id` is a new reading. The reference preparation `REF-P` may be reused for new bodies.

Returns arrive in the same call. Each native `bash` command is stopped after 15 minutes. There is no reserve to keep and no request limit. A long investigation is compacted automatically when its context nears the model's limit; keep forecasts, evidence IDs and conclusions in workspace files so they survive. `COVERAGE.md` is refreshed after every request with what you have spent and which states you have measured.

## Measuring

- **Forecast every measurement.** Each `request_simulation` freezes your active model's forecast for every quantity the request returns: density, and Hci, Br and (BH)max for a loop. A forecast row names the `specimen_id` and holds one object per quantity with `mean`, `sd`, `unit`, `status` and `measurement_stage` (the request's output stage; a forecast without a matching stage, or without a positive sd, is not compared with its return). A null mean with a reason abstains. Every row must include `density_g_cm3` and `grain_size_nm` objects, either of which may abstain. Each forecast is compared with its return in `inspect_feedback`; forecasts are not part of the grade. Stage a model (`stage_model`) and adopt it (`adopt_model`) before your first request: the owner's starting model abstains on density and grain size and gives no magnetic forecast.
- **Stages.** `as_sintered` → `after_first_anneal` (900 °C; 1, 2 or 4 h in the declared programs) → `after_second_anneal` (520 °C, 2 h). A request's `measurement_stage` is its `output_stage`. Annealing is irreversible: once a body is annealed, its earlier stage cannot be measured.
- **Identity.** A `piece_id` labels where a body is sampled; it is not a separately heat-treated coupon. `measurement_context.field_set_id` names an acquisition: a new ID is a new reading; repeating an unchanged request redelivers the same one.
- **Instruments.** Density readings report each method's nominal reading sd; this lot's actual scatter is not stated. `archimedes_saturated` reads bulk density without bias. `geometric` carries a fixed relative bias for this lot whose size is not stated. Unsealed `archimedes` carries open-pore bias and alone cannot establish true density. Loops: independent 2 % log reading errors on Hci and Br; the knee field (Hk) is read with its own error, including a demagnetization correction fixed per piece, and (BH)max comes from the read curve, normalized to each piece's own density; request `bhmax_MGOe` or `demagnetization_curve` to get (BH)max. Bodies of one preparation share that preparation's effect; bodies and pieces differ. Their scatter is not stated. SEM distribution readings (d90, coarse fraction) carry a width read from their own field set. The SPS log's geometry-derived densities carry a calibration error of about 2 %.
- **Executable controls.** Degassing completeness is an outcome of the program, not a control: this instrument runs `none` or the fitted 600 °C program (600–900 s). SPS temperature 650–950 °C, heating 10–100 °C/min, hold 60–1800 s, pressure 30–60 MPa, `pressure_full_at_c` 600–950 °C (at or below the peak, with `on_cooling` only), either current path, the first anneal at 900 °C for 1800–14400 s and the second at 520 °C for 7200 s are executable. States outside the declared routes are allowed for information, but only a declared route can be nominated.

## The record you start with

- This lot's reference body `REF-B1` (preparation `REF-P`), made by `fitted`, 4.0 µm, 850 °C, 300 s, 10 °C/min, 60 MPa, `on_cooling` (default ramp), `compact`, as sintered, with one geometric density and one full loop: `INITIAL_EVIDENCE.json`. It costs nothing and is not one of your 12 bodies. It is not annealed or measured again: a request that names `REF-B1` is refused.
- The physical campaign through P7: RPI bodies #1–#8, historical samples A–E and their loops, correspondence and images (`library/`, `read_evidence`). They are a different powder lot and process record. They do not bound this lot's routes.
- The supplied physics packages (`models/supplied/`; current C7.3) with their limits (`LABORATORY.md`).

## Scope

No method, model, itinerary or number of experiments is required. No physical execution or external message is authorized.
