# Harness

`setup.sh ROOT DATA` copies `tree/` to `ROOT`, adds the released environment data and container images from the dataset, fetches `dynamical-cli` at its pinned commit for the two native runtimes, and builds the frozen Python environments. `run.sh` then runs one world with a classical baseline or a language model. The main README lists the requirements.

`tree/` keeps the directory depth the runner expects: the runner, the transports and the world records locate components by relative position. The files most worth reading are under `tree/artifacts/studies/`:

| Path | Contents |
|---|---|
| `evaluation/env/CONTRACT.md` | The task every investigator receives, identical to `tasks/*/instruction.md` |
| `evaluation/env/LABORATORY.md` | The laboratory notes supplied with the task |
| `evaluation/env/prepare.py` | Prepares one episode for one world |
| `evaluation/env/run.mjs` | The episode runner, shared by every arm |
| `evaluation/env/run_verbatim_errors.mjs` | The same runner with a bridge that returns the legacy laboratory's errors verbatim, used for the replaced investigation |
| `evaluation/arms/classical/policy.py` | The design-of-experiments (`doe`) and Bayesian optimization (`bo`) baselines |
| `evaluation/arms/frontier/openrouter_transport.mjs` | The OpenRouter transport for the six evaluated language models |
| `evaluation/src/evaluation_world.py` | The evaluation world definition |
| `evaluation/vendor/` | The digital twin package `ndfeb_sps`, as installed in the frozen environments |
| `evaluation/python/native-runtime/` | The instrument runtime's adapter and bundle, laid over `dynamical-cli` by `setup.sh` |
| `bridge/` | The investigator-facing scientific bridge, the base investigation state and the provider pins |
| `agent-runtime/runtime/pi/` | The Pi 1.4.0 agent harness |

The grader and route truth are in `frozen/grader/`. The supplied physics model the agents use is `environment/models-agent/supplied/physics-model/` in the dataset. The native runtime binds the twin package's files by sha256 in `dynamical/bundle/ndfeb-sps/dependency_lock.json`, so a change to those files must be rebound there before the simulator accepts requests.
