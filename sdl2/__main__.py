"""The sdl2 command line: verify, unpack, reproduce and grade."""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path


def _sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify(data):
    data = Path(data)
    bad, n = [], 0
    for line in (data / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = data / rel
        n += 1
        if not p.is_file() or _sha(p) != digest:
            bad.append(rel)
    ok = not bad
    print(f"dataset: {n - len(bad)} of {n} files match MANIFEST.sha256")
    for rel in bad[:10]:
        print("  mismatch:", rel)
    return ok


def unpack(data):
    data = Path(data)
    archives = sorted((data / "traces").glob("e-*.tar.zst"))
    import zstandard
    for a in archives:
        dest = data / "traces" / a.name[:-len(".tar.zst")]
        if (dest / "SUMMARY.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with a.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd, tarfile.open(fileobj=rd, mode="r|") as tf:
            tf.extractall(dest, filter="data")
    print(f"{len(archives)} trace archives unpacked under {data / 'traces'}")


def grade(data, episode):
    """Grade one episode exactly as grade_all.py does."""
    from . import leaderboard
    leaderboard.install(Path(data) / "worlds/board-package")
    import grade_all
    episode = Path(episode).resolve()
    cfg = (episode / "simulation-magnetic/reference_world.json").read_bytes()
    key = hashlib.sha256(cfg).hexdigest()[:16]
    _, rows = grade_all.grade_world((key, [str(episode)], None))
    g = rows[0]
    (episode / "GRADE.json").write_text(json.dumps(g, indent=1, default=float) + "\n")
    reward = {"outcome": int(bool(g.get("Outcome"))), "supported": int(bool(g.get("Supported"))),
              "both": int(bool(g.get("Both")))}
    (episode / "reward.json").write_text(json.dumps(reward) + "\n")
    print(json.dumps(reward))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sdl2",
        description="Verify the dataset, reproduce the leaderboard and grade investigations.",
    )
    # Also accept --procs before the command; hidden from help.
    parser.add_argument("--procs", type=int, default=os.cpu_count() or 4, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "verify": "Check the dataset against its file manifest.",
        "unpack": "Extract the released investigation traces.",
        "reproduce": "Recompute route outcomes, regrade every investigation and compare with the leaderboard.",
        "grade": "Grade one investigation and write GRADE.json and reward.json.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("data", type=Path, metavar="DATA", help="local dataset directory")
        if name == "reproduce":
            command.add_argument(
                "out", type=Path, nargs="?", default=Path("sdl2-out"), metavar="OUT",
                help="output directory (default sdl2-out)",
            )
            command.add_argument(
                "--procs", type=int, default=argparse.SUPPRESS, metavar="N",
                help="worker processes (default host CPU count)",
            )
        elif name == "grade":
            command.add_argument(
                "episode", type=Path, metavar="EPISODE",
                help="investigation directory to grade",
            )
    args = parser.parse_args(argv)
    if args.command == "reproduce" and args.procs < 1:
        parser.error("--procs must be a positive integer")
    return args


def main(argv=None):
    sys.dont_write_bytecode = True  # leave the dataset unchanged
    a = parse_args(argv)
    if a.command == "verify":
        sys.exit(0 if verify(a.data) else 1)
    if a.command == "unpack":
        unpack(a.data)
        return
    if a.command == "grade":
        grade(a.data, a.episode)
        return
    from . import pipeline
    package = a.data / "worlds/board-package"
    r = pipeline.rebuild_leaderboard(a.data, a.out, package, a.procs)
    same_truth = sum(v["identical_to_released"] for v in r["truth"].values())
    print(f"truth: {same_truth} of {len(r['truth'])} worlds identical to the released tables")
    res = pipeline.compare(r["out_board"], a.data / "board")
    for f, v in res.items():
        print(f"  {f:44s} {v}")
    ok = same_truth == len(r["truth"]) and all(v.startswith("identical") for v in res.values())
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
