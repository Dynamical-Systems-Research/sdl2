# C7.2 — consistent anneal observations and piece evolution

October 23, 2026. Current supplied tool: `ndfeb-sps-physics-tool-20261023.8.2`; magnetic twin family `ndfeb_c7_campaign_v1`, lot `CAMPAIGN-C7.2-LOT`. C7.1, C7 and C6 remain separate, unchanged packages. This is a conditional scientific model, not experimental validation of a magnet.

The objective remains **Dy/Tb-free NdFeB with Hci ≥20 kOe AND (BH)max 45–48 MGOe together**, reproducibly and at the required operating temperature. Density is an intermediate quantity.

## What changed

1. **Evidence identity.** `condition_anneal()` now tracks original source and reading IDs. A different specimen reporting 7.58 is valid new evidence; a second ingestion of the same reading is refused. The two #8 calibration reading IDs are stored in the posterior metadata. Changing their values does not make them new observations.
2. **One piece evolution.** For each draw, the piece starts at body density plus an initial local offset. Its local porosity evolves through the complete anneal history. Calibration, `anneal_eig()`, `condition_anneal()` and the C7.2 twin use this same function. The twin previously added the initial offset after densification. This local-response approximation is explicit; no measured radial/axial density map validates it yet.
3. **Evidence available through P7.** Ten additional #8 as-sintered TIFFs, their full-frame PNGs and native FEI metadata are released. They add field coverage, not ten specimens or a new density/magnetic result. Numeric calibration remains P5. No new magnetic fit was made.

The anneal fit rebuild has identical numerical posterior arrays to C7.1; only provenance metadata changes. SPS density posterior, physics, microstructure and coercivity-link files are unchanged. Previous package behavior is retained in the owner runtime for historical replay. The C7.2 release retains the prior hidden seed, with a separate lot/version; no world was selected for a preferred answer.

## Reading and updating

```python
import ndfeb_physics_tool as tool
pm = tool.load()  # verifies TOOL_MANIFEST.json
route = {"T_peak_C": 800, "pressure_mode": "cooling", "P_MPa": 60,
         "anneal": "900", "anneal_hold_h": "as_#8"}
prediction = pm.predict(route, rho_observed=5.77)
```

`predict()` reports body-level stage density. Its `reading_g_cm3` summaries describe that density under the stated reading model, without a specific piece offset. Piece-level experiment design and updating integrate piece offsets; the twin fixes a persistent offset by specimen and piece ID. These are different sampling units, not interchangeable observations.

An update needs the **original** source ID, specimen ID and reading ID, plus measurement stage and piece identity. Use `piece_id: None` when unknown. Example schema only, not a new return:

```python
observation = {
    "source_id": "gmail:<original-message-id>", "specimen_id": "<panel-specimen-id>",
    "start": {"g_cm3": 5.77, "sd_rel": 0.03}, "program": "as_#8", "same_piece": True,
    "before": {"g_cm3": 5.77, "method": "geometric", "observation_id": "<original-before-id>",
               "measurement_stage": "as_sintered", "piece_id": "<piece-id>"},
    "after": {"g_cm3": 6.6, "method": "geometric", "observation_id": "<original-after-id>",
              "measurement_stage": "after_first_anneal", "piece_id": "<same-piece-id>"}
}
# Only once this is an actual new, traceable return:
# updated, diagnostics = pm.condition_anneal([observation])
```

Copies keep the original source/reading IDs. Do not rename a previously used baseline or a corrected reading to count it again. A correction to calibration requires an owner refit and a new version. This updater assumes independent specimens conditional on shared parameters; additional correlated observations on an already-conditioned specimen require a joint nuisance treatment. Inspect effective sample size before using a reweighted posterior. `start.sd_rel` is a declared starting-density prior, not another independent measurement.

For design, `anneal_eig()` accepts the same process/reading description without observed values or observation IDs. Supported density methods: unsealed `archimedes`, `archimedes_saturated` (dry, saturated, suspended weights), `geometric`, or `unspecified`. Unknown methods and pairing retain uncertainty. Saturated-weight handling must be qualified with RPI for reactive powder/porous samples; this is a model of a protocol, not a confirmed RPI capability.

## What the regenerated decision calculations mean

`build/DECISION_ANALYSIS.json` was regenerated from this package. Among the declared designs, matched before/after density on retained #1/#4 is more informative about anneal mechanism than #7 or an unspecified after-only reading. Under the assumed saturated-weight error model it resolves about **52–55% of the model's mechanism entropy**. Geometric readings give about 47–49%. These are conditional information calculations, not observed information gains or guaranteed experimental power.

The density criterion is **a sampled piece from a new 750/800 °C body reaching 7.2 g/cm³ after 900 °C / 2 h**. Under the current model, the 750 °C criterion is about 0.42. Hypothetical outcomes of the #1 experiment move that probability widely (q10/q50/q90 approximately 0.00/0.06/0.97). This is a distribution over future evidence, not a new result or an expected fall in success probability. Candidate-to-candidate prior differences reflect finite Monte Carlo samples. Starting-density, piece-gradient, hold and reading-error assumptions matter.

This does **not settle the material route**. A strong density gain would support investigating low-temperature SPS plus a load-free finish; a small gain would favor obtaining density elsewhere. Grain preservation, boundary behavior, texture and magnetic performance still determine whether either route serves the joint requirement. A matched magnetic comparison independently tests whether annealing adds or costs remanence on the current powder.

Three real densification mechanisms still fit the single #8 pair: LPS 0.373, CLOSED 0.349, CAPACITY 0.270. Artifact-only 0.008 is disfavoured within these priors and error models, not universally ruled out. Fine-powder rankings remain sensitive to unmeasured oxidation assumptions. The independent-grain isotropic bound does not model exchange-enhanced remanence. Operating-temperature magnetics remain extrapolations; no current-panel loop exists.

RPI must confirm retained-piece custody, piece pairing, density protocol and actual #8 heat program. A retained-piece anneal appears compatible with funded density/anneal work but scheduling and scope remain unconfirmed. Matched magnetic loops and alignment/powder-processing additions are separate, unquoted capabilities. No physical execution is authorized by this release.

## Reproduction and verification

- `INVENTORY.md`: input coverage and dependencies; inherited scientific assumptions.
- `build/c7_anneal_fit.py`: deterministic anneal calibration; `build/c7_density_update.py`: unchanged density build.
- `build/c7_decision_analysis.py`: design, magnetic comparison and joint-target calculations.
- `../../checks/c7-release-20261023/C72_MODEL_CHECKS.json`: identity, duplicate, calibration and actual twin matched-draw regressions.
- `../../checks/c7-release-20261023/README.md`: intake, runtime, end-to-end and preservation receipts.

CLOSED uses a numerically tabulated dose integral, not a symbolic exact solver; segmentation invariance is at rounding precision and numerical accuracy is checked against a fine-step reference. Passing tests establishes software consistency and provenance, not physical validity. No investigator has run on C7.2, and no training was performed.
