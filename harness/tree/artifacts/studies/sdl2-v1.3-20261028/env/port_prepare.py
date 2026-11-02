"""Build env/prepare_v13.py from v1.2's env/prepare_v12.py (not edited) with exact-text edits, each required to match the
stated number of times, and record the diff in env/prepare_v13.diff.

What changes: v1.3 worlds (sdl2_v13.draw from src/, never the shipped rows), this study's owner environment, v1.2's
base investigation used read-only, the 0.95 bar in the task objective, the capability text matched to CONTRACT.md
(drawn from the posterior, not its rows; the v1.3 readings), and a secret simulation seed separate from the world seed.
The base-rate sentence says only what the rung-2 ideal observer uses (no hints; Jarrod, Oct 28).

  python3 env/port_prepare.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "sdl2-v1.2-20261026/env/prepare_v12.py"
DST = HERE / "prepare_v13.py"

EDITS = [
    ("prepare_v12.py", "prepare_v13.py", 3),
    ("run_v12.mjs", "run_v13.mjs", 3),
    ("env/library-master and env/models-master/supplied", "env/library-agent and env/models-agent/supplied", 1),
    ("mounts env/library-master and models-master read-only here", "mounts env/library-agent and models-agent read-only "
     "here", 1),
    ('"""Prepare one SDL-2 v1.2 episode from the shared contract, the clean P7 base ledger and an owner world.',
     '"""Prepare one SDL-2 v1.3 episode from the shared contract, the clean P7 base ledger and an owner world (built from\n'
     '../sdl2-v1.2-20261026/env/prepare_v12.py by env/port_prepare.py; diff in env/prepare_v13.diff).', 1),
    ("import shutil\nimport sqlite3\n", "import shutil\nimport sqlite3\nimport sys\n", 1),
    ('STUDY = HERE.parent\n', 'STUDY = HERE.parent\nV12 = STUDY.parent / "sdl2-v1.2-20261026"\n', 1),
    ('BASE = HERE / "base-investigation"', 'BASE = V12 / "env/base-investigation"   # read-only: sqlite immutable, blobs hard-linked', 1),
    ('caps = c7.public_capabilities(world["base_world"])', 'caps = c7.public_capabilities(world["v12"]["base_world"])', 1),
    ('"Simulates one virtual powder lot. The hidden world is one posterior draw of the owner-built physics package C7.3 "\n'
     '        "(ndfeb-sps-physics-tool-20261023.8.3; its README states the data cutoff) with one of its two magnetic "',
     '"Simulates one virtual powder lot. The hidden world is drawn from the same posterior as the supplied physics "\n'
     '        "package C7.3 (ndfeb-sps-physics-tool-20261023.8.3; its README states the data cutoff), not from its stored "\n'
     '        "rows, with one of its two magnetic "', 1),
    ('"new reading noise per acquisition (field_set_id); independent 2% log reading errors on Hci and Br",',
     '"new reading noise per acquisition (field_set_id); independent 2% log reading errors on Hci and Br; the knee field "\n'
     '        "(Hk) is read with its own error, including a demagnetization correction fixed per piece",\n'
     '        "Density readings report each method\'s nominal reading sd; this lot\'s actual scatter is not stated",\n'
     '        "SEM distribution readings (d90, coarse fraction) carry a width read from their own field set; the SPS log\'s "\n'
     '        "geometry-derived densities carry a calibration error of about 2%",', 1),
    ('def world_for(panel, state, seed):\n    from ndfeb_sps.models import sdl2_v12 as v12\n',
     'def world_for(panel, state, seed):\n    sys.path.insert(0, str(STUDY / "src"))   # owner-side draws resolve owner/ '
     'through src/sdl2_v13.py\n    import sdl2_v13 as v13\n', 1),
    ('    world = v12.draw(LAB / "instruments/C7.3", seed, f"v12-adhoc-{seed}", meas, lot_id=LOT, drawn_at="2026-10-26")',
     '    world = v13.draw("eval", f"v13-adhoc-{seed}", meas, lot_id=LOT, drawn_at="2026-10-28", world_seed=seed)', 1),
    ('"calculations, the coercivity link or the C5 path, as this lot\'s magnetic response. Evaluation lots are selected so "\n'
     '        "that about half have a declared route meeting the requirement and the supplied model\'s favourite route passes in "\n'
     '        "only some. Instrument corrections:',
     '"calculations, the coercivity link or the C5 path, as this lot\'s magnetic response. About half of the lots in this "\n'
     '        "program have a declared route meeting the requirement. "\n'
     '        "Instrument corrections:', 1),
    ('for three bodies of a new preparation with probability >= 0.5; nominate it or restrict."',
     'for three bodies of a new preparation with probability >= 0.95; nominate it or restrict."', 1),
    ('seed=origin["seed"], world_config=world,', 'seed=sim_seed, world_config=world,', 1),
    ('    template = json.loads(', '    sim_seed = secrets.randbits(62)   # reading noise; independent of the world seed\n'
     '    template = json.loads(', 1),
    ('"version": "sdl2-v1.2-dev.1", "origin": origin,', '"version": "sdl2-v1.3-dev.1", "origin": origin, "sim_seed": sim_seed,', 1),
]


def main():
    old = SRC.read_text()
    text = old
    for a, b, n in EDITS:
        assert text.count(a) == n, (a[:80], text.count(a), n)
        text = text.replace(a, b)
    DST.write_text(text)
    diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True), "sdl2-v1.2-20261026/env/prepare_v12.py",
                                "sdl2-v1.3-20261028/env/prepare_v13.py")
    (HERE / "prepare_v13.diff").write_text("".join(diff))
    print(f"{len(EDITS)} edits -> {DST.name}")


if __name__ == "__main__":
    main()
