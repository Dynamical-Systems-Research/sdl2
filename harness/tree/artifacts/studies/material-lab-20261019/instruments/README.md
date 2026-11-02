# Supplied models — current entry point

**Current: `C7.3/`** (October 23, 2026, corrective). The shared physics tool is `ndfeb-sps-physics-tool-20261023.8.3`, and the campaign twin is `ndfeb_c7_campaign_v1` (lot `CAMPAIGN-C7.3-LOT`), which answers `request_simulation(simulation="magnetic")`.
- What changed, and what C7.3 and C7.2 correct: `C7.3/README.md`. Changes since C6: the SPS density update on #7/#8, the anneal-densification layer, stage-specific densities, one alignment input, rejection of unknown route keys, and joint-target probabilities on both magnetic paths. Inherited C7.1 corrections: segmentation-consistent annealing, one hold dependence on both magnetic paths, and `anneal_eig()` / `condition_anneal()` for anneal experiments.
- What each model consumes and what it is calibrated on: `C7.3/INVENTORY.md`.

**Superseded packages are archived byte for byte, not kept here** (since October 30), so searches of `models/` find only the current package. They are in `library/source_archive/instruments/physics/`:
- `C7.2/` (`ndfeb-sps-physics-tool-20261023.8.2`): its `condition_anneal()` keyed reading identity on the source string, so a reading entered under another source, evidence id or specimen label could count twice.
- `C7.1/` and `C7/` (`20261023.8.1`, `.8.0`): C7's CAPACITY and CLOSED anneal mechanisms depend on how a program is segmented; C7.1 corrected that.
- `C6-development/` (C6 v2, `20261018.7.0-dev`, density cutoff P3), whose twin family `ndfeb_c6_campaign_v1` and virtual specimens remain records, and the earlier tools C0–C5.

Use an archived package only to reproduce earlier results; earlier forecasts name the package they used. Packages load in one Python process (`import` each `ndfeb_physics_tool.py` under its own module name; each calls `load()`), so you can compare them directly. The packages are model hypotheses, not measurements, and not a replacement for a model you adopted yourself.

C7.2 added observation identity, duplicate rejection and consistent initial-piece evolution; C7.3 resolves that identity against the released ledger. C7.1 remains unchanged for replay. P7 adds images, not a numeric calibration. See C7.3/README.md for qualified density-route conclusions.
