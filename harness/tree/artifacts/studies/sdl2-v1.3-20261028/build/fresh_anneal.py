"""Fresh anneal-layer world stores (owner only; no network, no spend).

The shipped anneal_posterior.npz is a weighted resample of a 200,000-draw prior pool per mechanism that the shipped
build/c7_anneal_fit.py regenerates from the seed in the file's own metadata (20261023), so holding rows back from it
would not hide them. v1.3 worlds instead take anneal parameters from independent runs of the same fit (same priors,
same #8 likelihood, same nuisance integration, same resampling) under secret seeds: a statistically equivalent
posterior sample that no agent can regenerate. One store for train worlds, one for eval worlds.

  ../sdl2-v1.2-20261026/python/bin/python -I build/fresh_anneal.py train|eval

Writes owner/anneal_world_<split>.npz (the package's key layout and meta, with the seed removed) and appends the secret
seed to owner/ANNEAL_SEEDS.json (mode 0600)."""
import importlib.util
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np

V13 = Path(__file__).resolve().parents[1]
FIT = V13.parent / "sdl2-v1.2-20261026/env/models-master/supplied/C7.3/build/c7_anneal_fit.py"
OWNER = V13 / "owner"


def load_fit():
    spec = importlib.util.spec_from_file_location("c7_anneal_fit", FIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(split):
    assert split in ("train", "eval")
    f = load_fit()
    seed = secrets.randbits(63)
    rng = np.random.default_rng(seed)
    res = f.fit(rng)
    weights, logZ = f.mech_posterior(res)
    rng_s = np.random.default_rng([seed, 11])
    post = {m: f.resample(res[m], rng_s) for m in f.an.ANNEAL_MECHS}
    arrays = {f"post__{m}__{k}": post[m][k].astype(np.float64)
              for m in f.an.ANNEAL_MECHS for k in f.an.ANNEAL_PARAMS + f.an.MEASURE_PARAMS}
    meta = {"layer": "C7 anneal densification and density reading (v1.3 world store)", "version": "c7-anneal-1.2",
            "split": split, "mechanism_prior": f.MECH_PRIOR, "mechanism_weights": weights, "logZ": logZ,
            "ess": {m: res[m]["ess"] for m in res}, "n_prior_per_mechanism": f.N_PRIOR, "n_nuisance": f.N_NUIS,
            "n_store": f.N_STORE, "seed": None,
            "note": "independent run of build/c7_anneal_fit.py's fit under a secret seed (owner/ANNEAL_SEEDS.json)"}
    OWNER.mkdir(exist_ok=True)
    out = OWNER / f"anneal_world_{split}.npz"
    np.savez(out, __meta__=np.array(json.dumps(meta)), **arrays)
    os.chmod(out, 0o600)
    seeds_path = OWNER / "ANNEAL_SEEDS.json"
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    seeds[split] = str(seed)
    seeds_path.write_text(json.dumps(seeds) + "\n")
    os.chmod(seeds_path, 0o600)
    print(json.dumps({"split": split, "mechanism_weights": weights, "ess": meta["ess"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
