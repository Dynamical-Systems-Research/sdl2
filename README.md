# SDL-2

[![Research post](https://img.shields.io/badge/Research%20post-dynamicalsystems.ai-3D5A80)](https://dynamicalsystems.ai/blog/sdl-2)
[![Dataset on Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Dynamical-Systems/sdl2)
[![Release](https://img.shields.io/github/v/release/Dynamical-Systems-Research/sdl2?color=3D5A80)](https://github.com/Dynamical-Systems-Research/sdl2/releases/tag/v1.0.0)
[![Code license: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

SDL-2 measures whether AI agents can decide, from experiments they choose, whether a process will reliably meet a stated performance requirement. Each investigation takes place in a digital twin of magnet processing, starting with a new powder, an uncertain physics model and a fixed experimental budget. We grade whether the final decision is correct and whether the investigator’s own measurements support it.

The [research post](https://dynamicalsystems.ai/blog/sdl-2) presents the evaluation and findings.

This repository contains the evaluation harness, grader, both classical baselines, digital twin and instruments. The [dataset](https://huggingface.co/datasets/Dynamical-Systems/sdl2) contains the evaluation worlds, route outcomes, investigation records and scores.

## Reproduce the results

You need [uv](https://docs.astral.sh/uv/), `zstd` and about 17 GB of free disk space. Reproduction has been verified on Linux aarch64 and x86_64. It does not require Docker or access to a language model.

```bash
git clone https://github.com/Dynamical-Systems-Research/sdl2
cd sdl2
uvx --from huggingface_hub hf download Dynamical-Systems/sdl2 --repo-type dataset --local-dir data
uv run sdl2 verify data
uv run sdl2 unpack data
uv run sdl2 reproduce data out --procs 8
```

`verify` checks the released files against their recorded hashes. `unpack` extracts the investigation records. `reproduce` recomputes every route’s outcome with the frozen code, grades every investigation and rebuilds the leaderboard. A successful comparison ends with `REPRODUCED`.

The download includes the records needed to reproduce the results. You can inspect scores in `investigations.parquet` and route outcomes in `worlds/routes.parquet` before unpacking the traces.

<details>
<summary>Reproduction checks</summary>

The comparison checks every route outcome against the released truth tables and every leaderboard file against the released copy. It ignores the local episode path in `GRADES.jsonl` and compares the other files byte for byte.

Reproduction took about 10 minutes with `--procs 16` on aarch64. Set `--procs` to the CPUs available to your process. Its default uses the host CPU count, which can exceed a container’s allocation. macOS and Windows have not been tested.

</details>

## Scoring

Each of the 32 evaluation worlds supplies a new magnet powder. The investigator has 570 experiment credits, at most 12 new magnets and 8,064 candidate processing routes. It must nominate a route for which three magnets from a fresh batch all meet 10 kOe coercivity and 11 MGOe energy product with probability at least 0.95, or conclude that no route works.

- **Outcome** means the decision is correct in the hidden world and within the resource limits.
- **Supported** means the investigator’s own measurements meet a fixed evidence rule.
- **Both** requires a correct, supported decision.

<details>
<summary>Results by investigator</summary>

Each investigator ran once on every world. Counts below are out of 32 investigations per investigator. Exact binomial reference intervals accompany the Both rate. Model ordering is descriptive, with no pre-registered paired comparison significant after Holm correction.

| Investigator | Both | Exact 95% interval | Outcome | Supported |
|---|---|---|---|---|
| GPT-6 Astra | 16/32 | 31.9–68.1% | 23/32 | 18/32 |
| Claude Sonnet 5.5 | 15/32 | 29.1–65.3% | 20/32 | 16/32 |
| Gemini 4 Argon | 15/32 | 29.1–65.3% | 22/32 | 19/32 |
| Claude Opus 5.5 | 14/32 | 26.4–62.3% | 25/32 | 17/32 |
| DeepSeek V4.1 Flash | 9/32 | 13.7–46.7% | 16/32 | 12/32 |
| Bayesian optimization | 9/32 | 13.7–46.7% | 15/32 | 16/32 |
| Design of experiments | 9/32 | 13.7–46.7% | 24/32 | 10/32 |
| GLM-5.3 Flash | 8/32 | 11.5–43.4% | 16/32 | 13/32 |

</details>

<details>
<summary>Grading rules</summary>

A nomination is correct when the route’s true probability of meeting the fresh-batch requirement is at least 0.95. A conclusion that no route works is correct when no declared route reaches that probability. Exceeding 570 credits or 12 new magnets makes the decision incorrect.

A nomination is supported when the investigator has full magnetic curves on at least two magnets of the exact route and every magnetic reading of that route passes. A conclusion that no route works is supported when the investigator has full magnetic curves on at least one of the world’s 100 routes closest to passing and every such reading fails.

A magnetic reading passes when the stated noise model assigns at least 0.95 probability that the measured magnet meets the limit or limits tested, and fails at 0.5 or below. Full curves test both magnetic limits. Coercivity-only readings test coercivity and do not count toward the required full curves.

Route truth uses Monte Carlo over 64 to 2,048 simulated batches, stopping when the 95% Wilson interval excludes 0.95; a route whose typical magnet misses either limit by more than 3% counts as failing.

The frozen implementation is in [`grade_episode.py`](frozen/grader/grade_episode.py) and [`route_truth.py`](frozen/grader/route_truth.py).

</details>

## Run an investigation

Running the harness requires Linux aarch64. The released container images do not run natively on x86_64. The verified setup uses Docker with the NVIDIA container runtime, Node 24, uv, rsync, host user ID 1000, and ripgrep 15.2.0 and fd 10.4.2 in `~/.pi/agent/bin`.

Place the harness and episode directories outside any directory containing `.git`. The following example sets up the released harness, reruns a recorded Bayesian optimization investigation and grades it.

```bash
harness/setup.sh /var/tmp/sdl2 data
harness/run.sh /var/tmp/sdl2 /var/tmp/sdl2-episodes e-c74d169432 board-008 bo
uv run sdl2 grade data /var/tmp/sdl2-episodes/e-c74d169432
```

Use `doe` for the design-of-experiments baseline. A baseline’s seed comes from its episode name. Reusing a recorded name from `data/board/QUEUE_PHASE_D.json` reproduces that investigation’s decisions, tool calls and grade, apart from timestamps and run-directory names.

To run a language model, set `OPENROUTER_API_KEY` and replace `bo` with one of the model IDs below. These runs incur provider charges. New episodes remain as raw files, which `sdl2 grade` can read directly.

<details>
<summary>Model IDs</summary>

```text
anthropic/claude-opus-5.5
anthropic/claude-sonnet-5.5
openai/gpt-6-astra
google/gemini-4-argon
deepseek/deepseek-v4.1-flash-1017
z-ai/glm-5.3-flash
```

The harness pins each model to the provider used in the evaluation. Model availability and behavior can change. New runs use the released context. The original agents also had correspondence and article texts that are withheld from this release. The released context also renames the supplied model's folder, omits its earlier versions and edits the wording of the contract, laboratory notes and model notes; recorded traces show the original names.

</details>

## Repository contents

| Path | Contents |
|---|---|
| `sdl2/` | Commands to verify the dataset, unpack traces, reproduce the leaderboard and grade investigations |
| `frozen/` | The grader, route truth, digital twin and leaderboard scripts that graded the study. Files and identifiers were renamed and comments edited for release; `sdl2 reproduce` checks that every result is unchanged |
| `harness/` | Setup and run scripts, the original runner and both classical baselines |
| `tasks/` | Instructions and verifier commands for each evaluation world |

## Release scope

The dataset includes all 32 evaluation worlds with their hidden parameters and route outcomes, and all 256 investigations with their scores and released traces. RPI cleared its experimental data for this release. Personal names in the released records are replaced by `RPI` or `Dynamical`. Model traces are labeled AI-generated.

The release excludes RPI correspondence, the full text of two articles, provider-encrypted reasoning and signatures, and the 96 held-out worlds. It also omits large intermediate scratch files and pickle files from agent workspaces. Their hashes and the redactions are recorded in `WITHHELD.json`. The earlier versions of the supplied physics model, which the agents could also read, are not included; traces still name their files. The [dataset card](https://huggingface.co/datasets/Dynamical-Systems/sdl2#release-scope) describes these exclusions. They do not affect grading or reproduction of the published results.

## Evaluation notes

<details>
<summary>Study conditions and recorded issues</summary>

- **Requirement wording.** The task says “Bodies measured at 20 °C with Hci ≥ 10 kOe and (BH)max ≥ 11 MGOe”. Truth
  is judged on each magnet's true values, not on readings. An agent could read the requirement as applying to its
  noisy readings; this matters most in the four worlds with small performance margins below.
- **Small performance margins.** In 4 of the 16 feasible worlds, flagged before the run, every passing route clears the
  requirement by less than the instruments resolve, so a supported nomination needs readings above a typical
  magnet's. `data/board/SENSITIVITY.json` reports every investigator with and without them.
- **Replaced investigation.** One GLM-5.3 Flash episode (board-003) sent every request to the legacy laboratory, which hid
  its error, and decided with no data. It was rerun on a patched runner that returns that laboratory's errors verbatim
  (`env/run_verbatim_errors.mjs` in the harness); the rerun never reached the legacy laboratory. Both episodes score Outcome false
  and Both false. The original is in `data/replaced/`.
- **Bayesian optimization.** Its implementation sent a forbidden reference specimen in every first anneal batch, which
  failed, and its scheduler moved on rather than retrying. The score stands as run.
- **Route near the probability threshold.** board-016 is feasible through one route only, at P = 0.9526 after 2,048 batches; the
  pre-registered rule decides by the point estimate. Counting it infeasible would move Gemini 4 Argon from 15 to 14
  and DeepSeek V4.1 Flash from 9 to 10.
- **Operations.** Some commands reached time or memory limits. The released traces preserve these failures.
- **No `.git` above the episode directory.** The frozen bridge refuses to keep an episode's state below any directory
  that contains a `.git` entry, even an empty one, and the run fails at initialization with "private state must be
  outside version-controlled directories". Put the harness root and the episode directory somewhere like `/var/tmp`.
- **Records name host paths.** World records name the original simulation package by its original absolute path; the path and its
  hashes seed truth and readings, so it is kept and routed (`harness/sitecustomize.py`, `sdl2/leaderboard.py`).
- **Platforms.** Reproduction gives identical results on Linux aarch64 and x86_64 (CPython 3.13.3 to
  3.13.18, numpy 2.5.3, scipy 1.18.1); macOS and Windows were not run. Running a world needs aarch64.

</details>

## License and citation

Code is released under [Apache-2.0](LICENSE). The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Model outputs remain subject to their providers’ terms.

If you use SDL-2, its data or its harness, please cite the research post. [CITATION.cff](CITATION.cff) holds the same metadata for GitHub’s citation tool.

```bibtex
@article{barnes2026sdl2,
  author  = {Barnes, Jarrod},
  title   = {Teaching AI to Explore the Physical World},
  journal = {Dynamical Systems},
  year    = {2026},
  note    = {SDL-2},
  url     = {https://dynamicalsystems.ai/blog/sdl-2}
}
```
