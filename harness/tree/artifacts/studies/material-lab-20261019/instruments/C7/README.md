# Physics tool C7 — shared model release, October 23, 2026

C7 is the current shared physics model and campaign twin. C6 v2 (`../C6-development/`) is kept byte for byte, and so are its twin family and every earlier version. Your own adopted models are not replaced. Use C7, compare against it, or keep your own for a stated reason.

| | |
|---|---|
| Tool | `ndfeb-sps-physics-tool-20261023.8.0`, slice `C7-material-lab` (`TOOL_MANIFEST.json` binds every file) |
| Evidence cutoff | released evidence through P6 (October 22); densities through P5 (#8, Gmail `1a15becba0000b94`, 2026-10-21T00:46:20Z) |
| Twin | family `ndfeb_c7_campaign_v1`, lot `CAMPAIGN-C7-LOT`, `request_simulation(simulation="magnetic")` |
| Unchanged from C6 | `physics.py`, `forward.py`, `microstructure.py`; the coercivity link's core (only its tool adapter changed) |
| Not established | any current-panel magnetic value, the anneal mechanism, #8's anneal hold, density method and piece pairing |

```python
import ndfeb_physics_tool as tool          # run from this directory
pm = tool.load()                           # verified against TOOL_MANIFEST.json
pm.predict({"T_peak_C": 850, "pressure_mode": "cooling", "P_MPa": 60, "anneal": "900", "anneal_hold_h": "as_#8"},
           rho_observed=5.77)              # a retained 5.77 g/cm3 body through #8's (unconfirmed) program
pm.joint_target({"T_peak_C": 850, "pressure_mode": "cooling", "P_MPa": 60, "alignment_cos": 0.95,
                 "anneal": "900+520"})     # Hci >= 20 kOe and (BH)max >= 45 MGOe, both magnetic paths
```

## What changed, and on what evidence

**1. SPS density.**
- **Update.** The C5.0 density draws are reweighted on #7 (7.01, Archimedes) and on #8 as sintered (7.07, method not stated, given an extra 1 % sd). The two bodies share one new batch, `PREP-C850`; the timeline's degassing batch 3 is #7–#9.
- **Not used.** #8's after-900 reading is the anneal layer's observation, not this one's.
- **Effective sample size.** 780–1,030 distinct draws per density mechanism.
- **Check.** Reweighting on #7 alone reproduces the archived C5.1-prelim v2 forecast of #8: 7.039 [6.84, 7.30] against 7.038 [6.84, 7.28]. That is a consistency check, not a prediction.

| 850 °C panel route (g/cm³, q05 / median / q95) | C6 v2 | C7 |
|---|---|---|
| new batch | 6.54 / 7.26 / 7.52 | 6.45 / 7.12 / 7.44 |
| another body of batch 3 (#9) | — | 6.88 / 7.04 / 7.24 |

**2. Anneal densification** (`anneal.py`, `anneal_posterior.npz`, `build/c7_anneal_fit.py`). A load-free 900 °C step can densify a body. Four mechanisms are carried, each with its source:

| Mechanism | Physics | Investigators' special case | Weight after #8 |
|---|---|---|---|
| LPS | liquid-assisted, all porosity, first order | DeepSeek v7; Opus v2 M-LPS | 0.37 |
| CLOSED | closed porosity only; open compacts barely shrink | Opus v2 M-open; Qwen's pre-registered v1.7 | 0.35 |
| CAPACITY | at most c7_cap of porosity removable per anneal | Qwen v1.6 fixed increment | 0.27 |
| NULL | no densification; the change is a method, piece or reporting artifact | everyone's caveat | 0.008 (0.02–0.03 under wider nuisance priors) |

- **Nuisances integrated per draw:**
  - each stage's method (Archimedes vs geometric, 50/50);
  - same piece vs different piece (50/50);
  - per-piece gradients;
  - the TD denominator (7.58–7.70);
  - #8's hold (1, 2 or 4 h; weights 0.25/0.5/0.25).
- **What #8 identifies.** Only the dose (rate × hold; 900 °C dose ≥ about 1.6, q05).
- **Holds.** "As #8" predictions need no hold. Explicit holds (`anneal_hold_h`, 0.5–4 h, default 2 h as planned) use the declared #8-hold scenario.
- **The pair cannot rank LPS, CLOSED and CAPACITY.** They disagree below about 90 %TD:

| Retained body through #8's program (true density, g/cm³, median [q05, q95]) | LPS | CLOSED | CAPACITY | NULL |
|---|---|---|---|---|
| #1 (4.83, 750 °C) | 7.58 [7.08, 7.64] | 5.52 [4.88, 7.61] | 5.60 [5.15, 5.95] | 4.83 |
| #4 (5.77, 800 °C) | 7.58 [7.26, 7.64] | 6.25 [5.78, 7.62] | 6.55 [6.09, 6.90] | 5.77 |
| #7 (7.01, 850 °C) | 7.58 [7.49, 7.64] | 7.58 [7.45, 7.64] | 7.58 [7.31, 7.64] | 7.01 |

- **Retained 750/800 °C piece.** One same-piece, same-method before/after reading carries about 0.57 nats, half the mechanism entropy.
- **#7.** Its reading carries about 0.04 nats on mechanism: it tests only whether the densification replicates.

**3. Stage identity.** At an annealed route, `density_rel` is the body's density at that stage, with `density_rel_as_sintered` reported beside it. That stage density reaches four places:
- C5's Br, scaled by the body's density gain over historical E's own gain, because C5's anneal factor was fitted on E;
- the link's Br;
- the fracture feature factor;
- the twin's density, planarity and porosity readings.

The C5 grain law's 900 °C growth follows the hold. C5's magnetic anneal terms remain E's 2 h response at every hold.

**4. Inputs.**
- Unknown route keys raise, with the closest valid keys suggested.
- `degassing_completeness` is applied as `degas_completeness` and echoed.
- `alignment_cos` and `alignment_cos_theta` are one ⟨cos θ⟩ (0.5–0.95) that feeds both paths; giving both with different values raises.
- `predict()["route_resolution"]` records what was applied.
- `condition()` refuses an annealed-stage density or Br; annealing is the anneal layer's job.

**5. `joint_target()`.** Both magnetic paths run on one set of draws. Bodies of one preparation share everything they physically share:
- parameters, the batch effect and degassing state;
- the boundary-film magnetization J_GB and the link's epistemic draws;
- the anneal draw and the temperature scenario.

Body scatter comes from C5's fitted terms. P(all n bodies) = E[p(θ)ⁿ]. The independent product is reported only as a contrast. `assumptions` lists how the paths differ, and `structural` gives the Br²/4 bound.

**6. Twin readings.**
- `measurement_context.density_method`: `archimedes` (default) or `geometric`. Archimedes counts a world-fixed share of open porosity as solid; geometric carries a world-fixed bias.
- `piece_id`: the same piece at two stages shares its offset from the body.
- `surface: "fracture"` refuses polished-section readings.
- `normalization_density_g_cm3` changes Br and (BH)max, but not Hci.
- Magnetic distributions are corrected material quantities on RPI's DG11-3 basis (N = 0.36).
- `cooling_mode` is recorded but has no modeled effect.

## What the update says for decisions (conditional calculations; `build/DECISION_ANALYSIS.json`)

- **(BH)max needs alignment.** Isotropic bodies are capped near Br ≈ Js·ρ/2 ≈ 8 kG, so (BH)max ≤ about 16 MGOe at any coercivity. Every isotropic SPS route has P(joint) = 0 on both paths. That is a property of isotropic magnets, not of this instrument.
- **Coercivity is the binding leg, even with alignment declared.**
  - Aligned (⟨cos⟩ 0.95), 900 + 520 °C, 4 µm feedstock: P(one body meets both) is 0.05 % (C5) and 0.13 % (link). P(three bodies of one preparation) is about 0 on both paths.
  - Finer feedstock does not help in these models, because the declared C4 oxidation and hydrogen priors penalize it. No record constrains them. This is a limitation of the model, not evidence against fine powder.
- **A matched as-sintered/annealed magnetic comparison separates the two paths.** For #8's material after 900 °C:

| Ratio after/before, 900 °C, #8's material | C5 path | Coercivity link |
|---|---|---|
| Br | 0.93 [0.87, 1.00] (95 % of draws below 1) | 1.07 [1.05, 1.09] (1 % below 1) |
| Hci | 1.09 [1.01, 1.16] | 1.00 [0.81, 1.55] |
| (BH)max | 0.86 [0.76, 0.99] | 1.11 [1.00, 1.43] |

  - The Br difference is the one that matters: is historical E's anneal Br loss (8.0 → 7.0 kG) a property of annealing, or is the density gain dominant?
  - A paired loop read to about 2 % tells the two apart in about 93–99 % of draws.
  - This is a different uncertainty from the density mechanism, which the retained-piece anneal addresses.

## Limits that matter

- **Anneal layer.** One paired return, with every nuisance integrated and none fixed. Its mechanism weights are prior-dominated.
- **Coercivity.** No current-panel magnetometry exists. Both magnetic paths are anchored on the same historical E/DG11-3 records, so their agreement is not independent.
- **Prior-dominated levers.** Feedstock size (C4), degassing completeness, alignment, operating temperature and the link's coverage and J_GB all rest on priors. Treat results that depend on them as conditional.
- **Operating temperature.** The declared temperature scenario is unreliable for bodies whose room-temperature Hci is below about 5 kOe: the Kronmüller term can drive Hci(T) to zero.
- **Unchanged paths.** `eig()` remains the C5 path without anneal densification.

## Reproduce

From this directory:
```bash
python3 -B build/c7_anneal_fit.py .
```
```bash
python3 -B build/c7_density_update.py ../C6-development .
```
```bash
python3 -B build/c7_decision_analysis.py
```
Coverage and dependencies: `INVENTORY.md`. Verification receipts are on the operator side, in `checks/c7-release-20261023/`.
