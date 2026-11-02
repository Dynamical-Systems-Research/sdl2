# Coverage and dependency inventory — October 23, 2026 (C7.1)

The C7.1 corrections are listed in `README.md`. In the defects table below, the right-hand column describes C7.1.

This page lists what each shared model or virtual instrument consumes, its calibration cutoff, its main assumptions and what investigators actually receive. Package README: `README.md`. Primary records sit under the source archive (`library/source_archive/`) and `artifacts/rpi/`. Values are quoted from the records, not recomputed.

## Evidence the models can see

**Cutoff.** Released evidence runs through P6. Its newest files are three #8 after-900 °C fracture TIFFs, created on Drive 2026-10-21 20:33–20:34 UTC and released 2026-10-22. The newest RPI data message is Gmail `1a15becba0000b94` (2026-10-21 00:46:20 UTC), in thread `1a137e8fd3c4208f`.

**Access limitation, October 23.** Every Gmail and Drive connector call from this session failed at the host's tool hook. So this inventory could not check that thread or the Drive folder `[withheld:drive-folder-id]` for newer returns. No local file in the repository is newer than 2026-10-22.

| Body | Stage | Density, g/cm³ (as reported) | Method | What is not stated |
|---|---|---|---|---|
| #1–#3 (750 °C) | as sintered | 4.83, 4.78, 4.68 | mass/dimensions on polished pellets (RPI, Oct 12) | #3 and #5 disagree with their own mass/dimensions (5.47 and 5.17) |
| #4–#6 (800 °C) | as sintered | 5.77, 5.66, 5.63 | mass/dimensions | piece identity of the density, polished-section and fracture readings |
| #7 (850 °C) | as sintered | **7.01** (7.10 was a typo) | Archimedes | piece; its 900 °C anneal was completed on Oct 18, but no density or image has been received |
| #8 (850 °C) | as sintered | 7.07 (92 %TD as reported) | not stated | method, uncertainty, TD denominator, SPS log, mass/dimensions |
| #8 | after 900 °C | 7.58 (99 %TD as reported) | not stated | hold, atmosphere, cooling, piece pairing, method |

- **RPI's %TD figures imply a denominator of 7.59–7.70 g/cm³.** The tools' basis is 7.62.
- **Images.** There are 96 native SEM TIFFs, all fracture or lightly polished surfaces, with no EDS. For #8 that is 8 as-sintered and 3 after-900 °C images; two of the three annealed views overlap.
- **Magnetometry.** Historical only: A, C, D, E, E_1–3, E's 900 °C and 900 + 520 °C anneals, and the native DG11-3 loop. That loop was taken on a Quantum Design DynaCool VSM at 300 K, ±90 kOe, on a cube with a demagnetization correction of N = 0.36. **No current-panel loop exists.**

## Shared models and instruments

| Component | Consumes | Calibrated on / cutoff | Main assumptions | Investigators receive |
|---|---|---|---|---|
| C5 SPS density (D0–D3) | peak T, rate, dwell, pressure and load timing, cooling, degassing, batch (prep), feedstock | historical A–E, three S18 baselines, #1–#6 (C4 refit, cutoff P3). **C7 adds #7 and #8 as sintered, as batch PREP-C850** | densities relative to 7.62; Student-t scatter; one effect per powder batch; the reported #1–#6 values used as measured | C6 and C7 `predict()`; the twin's density readings |
| C5 coercivity and remanence (H0–H5, three load-timing explanations) | route, density, degassing state, grain, anneal key | historical A, C, D, E, E_900, E_900+520 (identical C0–C5) | isotropic; room temperature; C5 Br = Js·ρ·⟨cos⟩·(1+κ)·R(Hci)·anneal factor; squareness fitted on isotropic curves; Hci ceiling 37.5 kOe | `predict()`, `p_meets_spec()`, `eig()` |
| C4 feedstock layer | `feedstock_d50_um` | **declared, never fitted** | oxidation, hydrogen and grain-size penalties for fine powder | carried in every fine-powder magnetic result; **it decides the fine-powder ranking** |
| C5 SEM/BSE layer | C5 grain law, continuity W | declared; reweighted on three historical BSE classes | the grain law's 900 °C growth over 120 min; ratio about 1.26, against metrology's about 1.05 (an open issue) | `grain_size_sem_um`, `gb_phase_bse` |
| C6 microstructure layer | C5 latents, body density | cut posterior on owner metrology of #4–#7, E900, C, D, E (P4) | fracture features of bodies below about 95 %TD are particles, not grains; planar sections need at least 80 %TD | fracture intercept, SEM grain distribution, Nd-rich phase readings |
| C6 coercivity link v2 | stage grain, density, Nd-rich placement, chemistry offset, ⟨cos θ⟩ | absolute anchor on historical E (7.5 kOe); Phase 2a micromagnetic tables (48 loops); film magnetization J_GB ∈ {0, 0.6, 1.2} T with weights 0.26/0.65/0.08 | coverage prior anchored at D and E; the anneal re-wetting and the 520 °C gain are declared; body scatter not modelled | `predict()["coercivity_link"]`; every magnetic quantity from the twin |
| C5 joint probability (`p_meets_spec`) and goal EIG | the C5 path only | as C5 | one body; does not read the link; alignment scales Br only | `p_meets_spec`, `eig(goal=…)` |
| C6 twin, `ndfeb_c6_campaign_v1` | one hidden draw of the C6 package; manufacture, anneal, measure | as C6 | anneal steps of exactly 900 °C/7200 s and 520 °C/7200 s; **density unchanged by annealing**; isotropic; one reading-noise level | `request_simulation(simulation="magnetic")` until this release; no live investigator store holds a C6 body (fixtures only) |
| Legacy companion, `rpi_conditional_workflow_v1` | conditional workflow model | earlier campaign | separate historical simulator | `request_simulation` with no `simulation` key (DeepSeek holds 2 legacy specimens) |
| Tool slices C0–C4 | — | ledger-released | — | `library/tools/physics_tool_c0..c4`, usable through `compute` |

## Investigators' own post-#8 models

Each investigator's adopted model is preserved and was not replaced by this release.

- **Opus v2** (`model-11eb31a1cc8641d3ec6cbb53`): a bimodal anneal model. M-LPS leaves 10 % of the porosity; M-open adds +0.04 RD below 90 %TD.
- **Qwen v1.6** (`model-5d4546d756b04e4a9fb795ec`): a fixed +0.51 g/cm³, capped at 7.60, with a stated sd of 0.15. Qwen calls that sd knowingly overconfident.
- **DeepSeek v7** (`model-986af9f598ae4488bee6994a`): ρ + 0.90(0.995 − ρ), but starting from its own logistic in temperature, not from the released density.
- **Astra:** abstains on all of this.

All three anneal models fit the same single #8 pair at TD 7.62, so their agreement on #8 is built in. On the retained 750/800 °C pieces they differ by more than 1 g/cm³.

The magnetic models are not independent either:
- They share the historical E / DG11-3 anchor, the deck's anneal factors and the Kronmüller frame.
- Four separate raw reductions of DG11-3 agree. That agreement is independent arithmetic on one loop, not a new specimen.

## Defects found in C6 v2 and where C7 addresses them

| C6 v2 behaviour | Effect | C7 |
|---|---|---|
| `_route()` accepts any key (`r.update(route)`) | a misspelled `degassing_completeness` silently ran at the default | unknown keys raise with the closest route keys; the twin's `degassing_completeness` is an echoed alias |
| `alignment_cos` feeds C5 and `alignment_cos_theta` feeds the link | e.g. C5 Br 13.9 against link Br 7.4 at "0.9" | one ⟨cos θ⟩ feeds both paths; conflicting values raise |
| annealing never changes density (tool and twin) | contradicted by #8 (7.07 → 7.58) | anneal layer; stage densities everywhere they enter. C7.1: segmentation-consistent (C7's CAPACITY and CLOSED were not) |
| the link reads the as-sintered density at every stage | annealed Br is understated | stage density per stage |
| the twin accepts only 900 °C/7200 s | #8's unstated hold cannot be represented | 900 °C for 1800–14400 s; the "as_#8" program in the tool |
| one density reading per body, no method or piece | the method, piece and open-porosity problems are invisible | Archimedes or geometric reading, with a per-piece offset |
| polished-section readings with no surface notion | fracture-surface brightness was read as phase change (P6) | `surface: fracture` refuses section readings |
| twin magnetic limits said "apparent, without demagnetizing correction" | the link is anchored on RPI's corrected reduction | stated as corrected material quantities; Hci is independent of the correction; normalization density is stated |
| `p_meets_spec` is C5 only and treats bodies as one draw each | no compatible joint probability | `joint_target()`: both paths, shared J_GB and epistemic draws per preparation, E[pⁿ] |
| `condition()` would fit an annealed density as if sintered | silent stage error | refuses; `condition_anneal()` updates the anneal layer instead (C7.1) |
| `eig()` scores annealed stages without anneal densification (also in C7) | it shows no information about the anneal question | refuses those quantities; `anneal_eig()` on the anneal layer (C7.1) |
| the hold changed the link's coercivity but not C5's (C7) | inconsistent physical dependence | the hold-scaled grain enters both paths; non-grain anneal terms are E's 2 h response on both (C7.1) |
| section porosity and planarity: tool at the as-sintered density, twin at the stage density (C7) | two bases for one reading | both at the stage density (C7.1) |

**Unchanged in C7.1, with limits stated:**
- `eig()` stays on the C5 path. It now refuses what it cannot represent.
- Non-grain magnetic anneal terms are historical E's 2 h response at every hold, on both paths.
- The C4 feedstock layer remains declared. Its oxidation and hydrogen priors set the fine-powder ranking.
- The isotropic remanence of both paths is that of independent grains; exchange-enhanced remanence is not modelled.
