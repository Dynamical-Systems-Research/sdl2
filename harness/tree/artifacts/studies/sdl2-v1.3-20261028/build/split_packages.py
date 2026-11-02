"""SDL-2 v1.3 world split (owner only; no network, no spend).

v1.2 drew every hidden world at exact row indices of the posterior arrays that agents receive (models/supplied and the
library's physics tools), so any world-fixed number returned exactly is an index lookup. v1.3 reserves rows for worlds
and removes them from everything an agent can read.

For each posterior mechanism of the C7.3 posterior_tool.npz (density D0-D3, magnetic H0-H5), identical rows are grouped
(resampling repeats rows) and the groups are split by a secret permutation into shipped (half), train-world (a
quarter) and eval-world (a quarter) sets. Every shipped .npz (models and library, every package version, posterior and
prior groups) then loses each row whose values on the keys it shares with C7.3 equal a train- or eval-world row. The
anneal layer is not split here: v1.3 worlds draw anneal parameters from fresh secret-seeded fits (fresh_anneal.py),
because the shipped anneal store is a resample of a prior pool that its own build script regenerates.

  ../sdl2-v1.2-20261026/python/bin/python -I build/split_packages.py      (run from the v1.3 study directory)

Writes owner/WORLD_SPLIT.json (private: row indices per set), env/models-agent/ and env/library-agent/ (agent-visible
copies; unchanged files hard-linked to the v1.2 masters), and build/SPLIT_RECEIPT.json (counts and sha256 only)."""
import hashlib
import json
import os
import secrets
import shutil
import zipfile
from pathlib import Path

import numpy as np

V13 = Path(__file__).resolve().parents[1]
V12 = V13.parent / "sdl2-v1.2-20261026"
MODELS, LIBRARY = V12 / "env/models-master", V12 / "env/library-master"
OUT_MODELS, OUT_LIBRARY = V13 / "env/models-agent", V13 / "env/library-agent"
OWNER = V13 / "owner"
SOURCE = MODELS / "supplied/C7.3/posterior_tool.npz"
SHARES = {"shipped": 0.50, "train": 0.25, "eval": 0.25}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def groups_of(z):
    """{(group, mechanism): sorted parameter keys} of a posterior file (the __meta__ entry aside)."""
    out = {}
    for k in z.files:
        if k == "__meta__":
            continue
        g, m, p = k.split("__", 2)
        out.setdefault((g, m), []).append(p)
    return {gm: sorted(ps) for gm, ps in out.items()}


def rows(z, g, m, keys):
    """(K, len(keys)) float64 matrix of one mechanism's rows on the given keys (float16/32 cast exactly)."""
    cols = [np.asarray(z[f"{g}__{m}__{k}"], np.float64).reshape(len(z[f"{g}__{m}__{k}"]), -1) for k in keys]
    return np.ascontiguousarray(np.concatenate(cols, 1))


def row_hashes(mat):
    return [hashlib.sha256(r.tobytes()).hexdigest() for r in mat]


def split_source():
    z = np.load(SOURCE, allow_pickle=False)
    rng = np.random.default_rng(secrets.randbits(128))   # secret; the split itself is the record
    split, keysets = {}, {}
    for (g, m), keys in sorted(groups_of(z).items()):
        if g != "post":
            continue
        mat = rows(z, g, m, keys)
        h = row_hashes(mat)
        uniq = sorted(set(h))
        order = rng.permutation(len(uniq))
        n_ship = int(round(SHARES["shipped"] * len(uniq)))
        n_train = int(round(SHARES["train"] * len(uniq)))
        part = {}
        for rank, gi in enumerate(order):
            part[uniq[gi]] = "shipped" if rank < n_ship else "train" if rank < n_ship + n_train else "eval"
        split[m] = {s: [i for i, x in enumerate(h) if part[x] == s] for s in SHARES}
        keysets[m] = keys
    return z, split, keysets


def world_rows(z, split, keysets):
    """{mechanism: {key tuple: set of row hashes}} for train- and eval-world rows, computed lazily per key subset."""
    cache = {}

    def hashes(m, keys):
        ck = (m, tuple(keys))
        if ck not in cache:
            idx = split[m]["train"] + split[m]["eval"]
            cache[ck] = set(row_hashes(rows(z, "post", m, keys)[idx]))
        return cache[ck]
    return hashes


def compressed(path):
    with zipfile.ZipFile(path) as zf:
        return any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())


def strip_file(path, out_path, hashes, keysets):
    """Write out_path without any row (any group) that equals a world row on the keys it shares with C7.3."""
    z = np.load(path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    report = {}
    for (g, m), keys in groups_of(z).items():
        if m not in keysets:
            continue
        common = sorted(set(keys) & set(keysets[m]))
        if not common:
            continue
        K = len(z[f"{g}__{m}__{keys[0]}"])
        mat = rows(z, g, m, common)
        bad = [i for i, x in enumerate(row_hashes(mat)) if x in hashes(m, common)]
        if not bad:
            continue
        keep = np.setdiff1d(np.arange(K), bad)
        for k in keys:
            a = z[f"{g}__{m}__{k}"]
            arrays[f"{g}__{m}__{k}"] = a[keep] if a.shape[:1] == (K,) else a
        report[f"{g}/{m}"] = {"rows": K, "removed": len(bad), "common_keys": len(common), "all_keys": len(keys)}
    if not report:
        return None
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()   # break the hard link to the master before writing
    (np.savez_compressed if compressed(path) else np.savez)(out_path.with_suffix(""), **arrays)
    return report


def mirror(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)   # hard links; changed files are rewritten below


def main():
    z, split, keysets = split_source()
    hashes = world_rows(z, split, keysets)
    OWNER.mkdir(exist_ok=True)
    os.chmod(OWNER, 0o700)
    (OWNER / "WORLD_SPLIT.json").write_text(json.dumps(
        {"source": str(SOURCE), "source_sha256": sha(SOURCE), "shares": SHARES,
         "note": "row indices into the source C7.3 posterior_tool.npz; identical rows share a set",
         "split": split}, indent=0) + "\n")
    os.chmod(OWNER / "WORLD_SPLIT.json", 0o600)
    mirror(MODELS, OUT_MODELS)
    mirror(LIBRARY, OUT_LIBRARY)
    receipt = {"source_sha256": sha(SOURCE), "shares": SHARES,
               "rows": {m: {s: len(v) for s, v in d.items()} for m, d in split.items()}, "files": {}}
    for master, out in ((MODELS, OUT_MODELS), (LIBRARY, OUT_LIBRARY)):
        for path in sorted(master.rglob("*.npz")):
            target = out / path.relative_to(master)
            rep = strip_file(path, target, hashes, keysets)
            entry = {"master_sha256": sha(path), "agent_sha256": sha(target), "changed": rep is not None}
            if rep:
                entry["removed"] = rep
                man = target.parent / "TOOL_MANIFEST.json"
                if man.exists():
                    m = json.loads(man.read_text())
                    for f in m.get("files", {}).values():
                        if f.get("name") == path.name:
                            f["sha256"] = entry["agent_sha256"]
                    m["agent_release"] = ("v1.3 agent copy: posterior rows reserved for hidden worlds are removed; the "
                                          "remaining rows are an unbiased subset of the same posterior")
                    man.unlink()
                    man.write_text(json.dumps(m, indent=2) + "\n")
                    entry["manifest_sha256"] = sha(man)
            receipt["files"][str(path.relative_to(V12))] = entry
    (V13 / "build/SPLIT_RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1))
    for k, v in receipt["files"].items():
        print(("CHANGED " if v["changed"] else "same    ") + k, json.dumps(v.get("removed", {}))[:400])


if __name__ == "__main__":
    main()
