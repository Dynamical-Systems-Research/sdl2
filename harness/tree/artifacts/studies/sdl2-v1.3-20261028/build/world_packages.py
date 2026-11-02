"""Owner world packages for v1.3 (owner only; never mounted for an agent).

owner/package-world-<split>/ holds the C7.3 package files with:
  posterior_tool.npz   the original, all rows (worlds index only their split's reserved rows; owner/WORLD_SPLIT.json)
  anneal_posterior.npz the split's secret-seeded refit (owner/anneal_world_<split>.npz) for the post group, and the
                       shipped prior group (prior draws never define a world)
  TOOL_MANIFEST.json   re-issued with the new anneal sha256, so the package verifies on every load

Also writes build/NOMINAL_SD.json: the method-level density reading sd reported by v1.3 (posterior RMS over the shipped
anneal sample, mechanism-weighted).

  ../sdl2-v1.2-20261026/python/bin/python -I build/world_packages.py
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

V13 = Path(__file__).resolve().parents[1]
MASTER = V13.parent / "sdl2-v1.2-20261026/env/models-master/supplied/C7.3"
OWNER = V13 / "owner"
RHO_TD = 7.62


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def nominal_sd():
    z = np.load(MASTER / "anneal_posterior.npz", allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    w = meta["mechanism_weights"]
    out = {}
    for par in ("c7_sd_geo", "c7_sd_arch"):
        ms = sum(w[m] * float(np.mean(np.asarray(z[f"post__{m}__{par}"], float) ** 2)) for m in w)
        out[par] = {"rms_g_cm3": float(np.sqrt(ms) * RHO_TD)}
    return out


def build(split):
    dst = OWNER / f"package-world-{split}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in MASTER.iterdir():
        if f.is_file() and f.name not in ("anneal_posterior.npz", "TOOL_MANIFEST.json"):
            shutil.copy2(f, dst / f.name)
    fresh = np.load(OWNER / f"anneal_world_{split}.npz", allow_pickle=False)
    shipped = np.load(MASTER / "anneal_posterior.npz", allow_pickle=False)
    meta = json.loads(str(shipped["__meta__"]))
    fmeta = json.loads(str(fresh["__meta__"]))
    meta.update(mechanism_weights=fmeta["mechanism_weights"], logZ=fmeta["logZ"], ess=fmeta["ess"], seed=None,
                v13_world_store=fmeta["note"], split=split)
    arrays = {k: fresh[k] for k in fresh.files if k.startswith("post__")}
    arrays.update({k: shipped[k] for k in shipped.files if k.startswith("prior__")})
    np.savez(dst / "anneal_posterior.npz", __meta__=np.array(json.dumps(meta)), **arrays)
    man = json.loads((MASTER / "TOOL_MANIFEST.json").read_text())
    man["files"]["anneal_posterior"]["sha256"] = sha(dst / "anneal_posterior.npz")
    man["v13_world_package"] = f"owner world package ({split}); anneal layer from a secret-seeded refit"
    (dst / "TOOL_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    for f in dst.iterdir():
        os.chmod(f, 0o600)
    os.chmod(dst, 0o700)
    return {"dir": str(dst.relative_to(V13)), "anneal_sha256": sha(dst / "anneal_posterior.npz"),
            "posterior_sha256": sha(dst / "posterior_tool.npz"), "mechanism_weights": fmeta["mechanism_weights"]}


def main():
    rec = {"nominal_sd": nominal_sd(), "packages": {s: build(s) for s in ("train", "eval")},
           "shipped_anneal_mechanism_weights": json.loads(str(np.load(MASTER / "anneal_posterior.npz")["__meta__"]))[
               "mechanism_weights"]}
    (V13 / "build/NOMINAL_SD.json").write_text(json.dumps(rec["nominal_sd"], indent=1) + "\n")
    (V13 / "build/WORLD_PACKAGES.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
