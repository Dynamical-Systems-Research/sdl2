# Laboratory

## Tools

- **Native Pi tools** (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) run in a disposable sandbox on your workspace. Bash has no network access; the literature and records this investigation uses are in `library/` (`read_sources`, `read_evidence`). The sandbox has 6 CPUs and 16 GB of memory; a command that exceeds the memory is killed. Its `/proc` reports the host machine's processors and memory, not these limits. A process left running in the background is not stopped by the 15-minute command limit; it ends with the investigation.
- **Scientific tools.** `request_simulation(request_id=..., simulation="magnetic", rationale=..., request={...})` makes, anneals and measures virtual bodies of this lot; the request names `"lot_id": "SDL2-LOT"`. `preview_request(request=..., simulation="magnetic")` checks a request and shows the forecast it would freeze, without running it. `inspect_feedback(request_id="magnetic:<request_id>")` compares that request's frozen forecast with its returns. `stage_model` and `adopt_model` set the model whose forecasts are frozen. `compute` runs recorded Python in an offline container (`inspect_state` section `compute_resources` describes it). `read_evidence`, `read_artifact`, `view_image`, `read_sources` and `inspect_state` read the record. `record_decision` with `stop` ends the investigation. `compact_context` compacts your context at a tool boundary, keeping the notes you pass it.
- A request without `simulation="magnetic"` reaches a separate legacy simulator of an older historical lot. Its returns are not this lot and are not part of this investigation. `submit_batch` records a physical proposal for review; nothing physical runs here.
- This model route reads text only: an image arrives as a placeholder. Inspect image files numerically with `compute` or `bash`.
- A refusal from the virtual instrument means that state is outside what it models, not that it is physically impossible.

## Request grammar

A new body:

```json
{"request_id": "MY-ID", "simulation": "magnetic", "rationale": "why this body",
 "request": {"origin": "synthetic", "operation": "manufacture", "lot_id": "SDL2-LOT",
  "preparations": [{"preparation_id": "PREP-A", "lot_id": "SDL2-LOT",
                    "conditions": {"method": "vacuum_degassing", "temperature_c": 600, "hold_time_s": 900, "feedstock_d50_um": 4.0}}],
  "specimens": [{"specimen_id": "BODY-A1", "preparation_id": "PREP-A", "cycle_id": "CYCLE-A1",
                 "conditions": {"temperature_c": 850, "heating_rate_c_min": 10, "hold_time_s": 300, "pressure_mpa": 60,
                                "pressure_schedule": "on_cooling", "mass_g": 0.9, "diameter_mm": 10,
                                "measurement_context": {"piece_id": "P1", "density_method": "geometric", "field_set_id": "ACQ-A1-1"}},
                 "measurements": ["density_g_cm3", "hci_kOe", "br_kGs", "bhmax_MGOe"]}]}}
```

For no degassing, the preparation conditions are `{"method": "none", "feedstock_d50_um": 4.0}`. An omitted `current_path` is `compact`; `pressure_full_at_c` is stated only with `on_cooling`. Every manufacture request lists each preparation it uses in `preparations`, including an existing one (repeat its `preparation_id` and identical conditions); a preparation is charged once, when it first appears.

An anneal or a later measurement of an existing body names the same specimen, preparation and cycle IDs, omits manufacture controls, and states `input_stage` and `output_stage`. An anneal adds `conditions.anneal_steps`, for example `[{"stage": "after_first_anneal", "temperature_c": 900, "hold_time_s": 7200, "heating_rate_c_min": 10}]` (the declared programs use 3600, 7200 or 14400 s), or the 520 °C step `{"stage": "after_second_anneal", "temperature_c": 520, "hold_time_s": 7200, "heating_rate_c_min": 10}`. A measurement uses `"operation": "measure"` with `input_stage` equal to `output_stage`. The bridge supplies each body's executed history; do not write `prior_processing_conditions`. Every request, an anneal included, names a nonempty `measurements` list for each specimen. Each density or loop reading needs a `measurement_context.field_set_id`.

## Supplied models

`models/supplied/physics-model/` is the live campaign's current physics package. The package directories are read-only; use it from anywhere with `PYTHONPATH=models/supplied/physics-model python3 ...` and `import ndfeb_physics_tool as tool; pm = tool.load()`, and write your own scripts and models elsewhere in the workspace (for example `work/` or `models/`). It reports two magnetic calculations, the coercivity link and the C5 path, on one set of draws (`joint_target(route, spec={"Hci_kOe_min": 10, "BHmax_MGOe_min": 11})`). What the package does at each declared control, the same for every lot:

| Control or quantity | What the package rests on |
|---|---|
| `preparation` | Magnetic module fitted on historical bodies made with and without degassing; the degassed state is the fitted 600 °C program. No other degassing program is modelled. |
| `feedstock_d50_um` 3.0, 2.0 | Every record used 4 µm feedstock. Finer feedstock rests on declared priors on oxidation and hydrogen, never fitted to any record; they set the package's fine-powder predictions. |
| `temperature_c`, `hold_time_s` | Density fitted on nine records made at 650–950 °C with 5–10 min holds; a 900 s hold is outside it. The magnetic module is fitted only on bodies made at 850 and 950 °C at 97–99 %TD, so its magnetic values at 750–800 °C or at lower density are extrapolations. |
| `heating_rate_c_min` | Density and magnetic records span 10–100 °C/min. The rate also scales the current-path dose term when load is applied in heating. |
| `pressure_mpa` | Density fitted at 50–60 MPa; 40 MPa is outside it. |
| `pressure_schedule`, `pressure_full_at_c` | Fitted records have load on heating or on cooling; none has load from the start of the dwell (`on_dwell`). With load on cooling, the default ramp is about 0.3 min (current RPI traces); full load at 700 °C is the slower ramp of historical E's recorded program. Densification and hydrogen release read the load schedule. |
| `current_path` | Calibrated only for a current-carrying compact (conductive graphite die); `insulated` is a declared scenario. It enters only the load-timing current dose term, which is zero while the compact carries no load, so with load on cooling the package gives the same body for either path. |
| `anneal` | Densification calibrated on one pair (#8, 7.07 → 7.58 g/cm³). Three mechanisms fit it about equally and disagree below about 90 %TD; an artifact explanation is disfavoured. The 900 °C hold scales stage density through that layer and grain through the hold-scaled grain law (D³ linear in time), on both magnetic calculations. Every other magnetic anneal term is historical sample E's 2 h response at every hold; the 520 °C gain is a literature value. |
| Hci, Br, (BH)max | The two paths disagree on annealed remanence (link: Br follows the density gain; C5: carries E's anneal loss). No loop exists for any current-panel body. For independent isotropic grains (BH)max is capped near Br²/4, about 16–19 MGOe in these models; exchange-enhanced remanence is not modelled. |

These are statements about the model, not about this lot. Your own adopted model is yours to build, test or replace.

## Physical record

`library/` holds the physical campaign through P7 as files (`library/README.md`): released RPI records in `evidence/` (start with `evidence/OVERVIEW.md`), historical samples and loops in `historical/`, and dated correspondence in `context_inputs/`. The same records are in `read_evidence`. If a file and a tool return differ, the tool return is the record.
