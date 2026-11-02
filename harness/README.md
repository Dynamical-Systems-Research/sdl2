# Harness

`setup.sh ROOT DATA` copies `tree/` to `ROOT`, adds the released environment data and container images from the dataset, and builds the frozen Python environments. `run.sh` then runs one world with a classical baseline or a language model. The main README lists the requirements.

`tree/` is the study's directory layout, copied file for file. The runner and the world records locate components by these paths, so the layout is kept as it ran. The files most worth reading are under `tree/artifacts/studies/`:

| Path | Contents |
|---|---|
| `sdl2-v1.3-20261028/env/CONTRACT.md` | The task every investigator receives, identical to `tasks/*/instruction.md` |
| `sdl2-v1.3-20261028/env/LABORATORY.md` | The laboratory notes supplied with the task |
| `sdl2-v1.3-20261028/env/prepare_v13.py` | Prepares one episode for one world |
| `sdl2-v1.3-20261028/env/run_v13.mjs` | The episode runner, shared by every arm |
| `sdl2-v1.3-20261028/arms/classical/policy_v13.py` | The design-of-experiments (`doe`) and Bayesian optimization (`bo`) baselines |
| `sdl2-v1.3-20261028/grader/` | The grader and route truth, identical to their copies in `frozen/grader/` |
| `sdl2-v1.3-20261028/src/sdl2_v13.py` | The evaluation world definition |
| `sdl2-v1.3-20261028/vendor/` | The digital twin package `ndfeb_sps`, as installed in the frozen environments |
| `sdl2-v1.3-20261028/python/native-runtime/` | The instrument runtime, laid over a pinned checkout of `dynamical-cli` by `setup.sh` |
| `sdl2-v1.2-20261026/arms/frontier/` | Model transport and provider pins |
| `material-lab-20261019/instruments/` | The supplied physics models (C7.3 is current) |
| `material-route-swarm-design-20261019/runtime/pi140/` | The Pi 1.4.0 agent harness |

Several files have a `port_*.py` script and a `.diff` beside them. These record how the file was built from its version in the previous study.
