"""Build leaderboard/leaderboard_v13.py (Phase C) from v1.2's leaderboard/leaderboard.py (not edited) with exact-text
edits, each required to match once, and record the diff in leaderboard/leaderboard_v13.diff.

What changes:
  - the runner and the preparer are v1.3's (env/run_v13.mjs, env/prepare_v13.py), and so is the sandbox name;
  - episodes live under /lab with random names, for every split (D12);
  - the board arms are D14's: Opus 5.5, Sonnet 5.5, GPT-6 Astra, Gemini 4 Argon, DeepSeek V4.1 Flash 1017 and GLM-5.3
    Flash, plus BO and DOE;
  - two queues:
    - SDL2_PANEL=board (the default) builds QUEUE.json from owner/BOARD_PANEL.json with every arm (C1);
    - SDL2_PANEL=heldout builds QUEUE_HELDOUT.json from owner/HELDOUT_PANEL.json with the DeepSeek V4.1 Flash 1017 base
      only, through OpenRouter (D4, C2). It is graded, never mined.
Unchanged: the queue order rule, sharding, stall detection, resume from the stopping point, and the checked compression of
passed episodes.

  python3 leaderboard/port_leaderboard.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "sdl2-v1.2-20261026/leaderboard/leaderboard.py"
DST = HERE / "leaderboard_v13.py"

EDITS = [
    ('"""The sealed leaderboard: every arm once on every sealed state',
     '"""SDL-2 v1.3 Phase C (built from ../sdl2-v1.2-20261026/leaderboard/leaderboard.py by leaderboard/port_leaderboard.py;\n'
     'diff in leaderboard/leaderboard_v13.diff). SDL2_PANEL=board (default): every D14 arm once on each of the 32 board\n'
     'states (C1); SDL2_PANEL=heldout: the DeepSeek V4.1 Flash 1017 base on the 96 held-out states (C2, D4). Episodes run\n'
     'under /lab (D12). The v1.2 text follows for the record.\n'
     'The sealed leaderboard: every arm once on every sealed state'),
    ('QUEUE = HERE / "QUEUE.json"',
     'PANEL = os.environ.get("SDL2_PANEL", "board")   # v1.3: board (C1) or heldout (C2)\n'
     'assert PANEL in ("board", "heldout"), PANEL\n'
     'QUEUE = HERE / ("QUEUE.json" if PANEL == "board" else "QUEUE_HELDOUT.json")'),
    ('EPISODES = HERE / "episodes"', 'EPISODES = Path("/lab")   # v1.3: every split under /lab, random names (D12)'),
    ('          "deepseek/deepseek-v4.1-flash-1017"]',
     '          "deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3-flash"]   # v1.3: D14 adds GLM-5.3 Flash'),
    ('ORDER = ["bo", "doe", "anthropic/claude-opus-5.5", "google/gemini-4-argon", "deepseek/deepseek-v4.1-flash-1017",',
     'ORDER = ["bo", "doe", "anthropic/claude-opus-5.5", "google/gemini-4-argon", "deepseek/deepseek-v4.1-flash-1017",\n'
     '         "z-ai/glm-5.3-flash",'),
    ('states = [st["name"] for st in json.loads((STUDY / "owner/SEALED_PANEL.json").read_text())["states"]]',
     'states = [st["name"] for st in json.loads((STUDY / f"owner/{PANEL.upper()}_PANEL.json").read_text())["states"]]'),
    ('for run in sorted(ARMS, key=lambda r: ORDER.index(label(r))):',
     'arms = ARMS if PANEL == "board" else [{"arm": "frontier", "model": "deepseek/deepseek-v4.1-flash-1017"}]   # v1.3: C2\n'
     '    for run in sorted(arms, key=lambda r: ORDER.index(label(r))):'),
    ('QUEUE.write_text(json.dumps({"panel": "sealed", "order_seed": seed, "entries": entries}, indent=1) + "\\n")',
     'QUEUE.write_text(json.dumps({"panel": PANEL, "order_seed": seed, "entries": entries}, indent=1) + "\\n")'),
    ('    QUEUE.write_text(json.dumps({"panel": PANEL, "order_seed": seed',
     '    # v1.3: the policies first, then each world\'s model arms together, longest first: the order v1.2\'s QUEUE.json ran\n'
     '    # in after October 26, when an arm-blocked order saturated the CPU (v1.2 leaderboard/README.md, "Order").\n'
     '    entries.sort(key=lambda e: (e["run"]["arm"] != "policy", 0 if e["run"]["arm"] == "policy" else e["order"],\n'
     '                                ORDER.index(e["label"]), e["order"]))\n'
     '    for i, e in enumerate(entries):\n'
     '        e["index"] = i\n'
     '    QUEUE.write_text(json.dumps({"panel": PANEL, "order_seed": seed'),
    ('print(f"{len(entries)} entries: {len(states)} states x {len(ARMS)} arms")',
     'print(f"{len(entries)} entries: {len(states)} states x {len(arms)} arms")'),
    ('"run_v12.mjs"', '"run_v13.mjs"'),
    ('"SDL2_V12_EPISODES": str(EPISODES)', '"SDL2_V13_EPISODES": str(EPISODES)'),
    ('"ndfeb-sdl2-ep-" + e["name"]', '"ndfeb-sdl2v13-ep-" + e["name"]'),
    ('"prepare_v12.py"', '"prepare_v13.py"'),
    ('EPISODES.mkdir(exist_ok=True)', 'assert EPISODES.is_dir(), "/lab is the episode root (D12)"'),
]


def main():
    old = SRC.read_text()
    text = old
    for a, b in EDITS:
        assert text.count(a) == 1, (a[:90], text.count(a))
        text = text.replace(a, b)
    DST.write_text(text)
    diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True), "sdl2-v1.2-20261026/leaderboard/leaderboard.py",
                                "sdl2-v1.3-20261028/leaderboard/leaderboard_v13.py")
    (HERE / "leaderboard_v13.diff").write_text("".join(diff))
    print(f"{len(EDITS)} edits -> {DST.name}")


if __name__ == "__main__":
    main()
