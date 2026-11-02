"""Build arms/frontier/openrouter_transport.mjs from v1.2's (not edited) with exact-text edits, each required to match the
stated number of times, and record the diff in arms/frontier/openrouter_transport.diff (A1/A9, D14).

What changes:
  - GLM-5.3 Flash is admitted. D14 (Jarrod, Oct 28) puts it on the v1.3 board. v1.2 dropped it from its release list
    on Oct 26, when Grok 4.7 replaced it, so v1.2's transport refuses it ("Exact requested model required"). Its PIN,
    catalog row and endpoint snapshot are v1.2's, recorded Oct 26 (Z.AI, fp8).
  - The PINs and snapshots are read from v1.2's arms/frontier/providers/, read-only. This copy sits at the same depth,
    so its pi140 import resolves to the same file.

  python3 arms/frontier/port_transport.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "sdl2-v1.2-20261026/arms/frontier/openrouter_transport.mjs"
DST = HERE / "openrouter_transport.mjs"
V12P = "path.join(chat.REPO, 'artifacts/studies/sdl2-v1.2-20261026/arms/frontier/providers')"

EDITS = [
    ("/** One OpenRouter route for every frontier model of the SDL-2 release",
     "/** v1.3 copy (arms/frontier/port_transport.py; diff in openrouter_transport.diff): admits z-ai/glm-5.3-flash (D14) and\n"
     " * reads v1.2's PINs and snapshots read-only.\n"
     " * One OpenRouter route for every frontier model of the SDL-2 release", 1),
    ("const HERE = path.dirname(fileURLToPath(import.meta.url));\n",
     "const HERE = path.dirname(fileURLToPath(import.meta.url));\n"
     f"const V12_PROVIDERS = {V12P};   // v1.3: v1.2's PINs and frozen snapshots, read-only\n", 1),
    ("path.join(HERE, 'providers')", "V12_PROVIDERS", 5),
    ("  'deepseek/deepseek-v4.1-flash-1017': V12_PROVIDERS,\n",
     "  'deepseek/deepseek-v4.1-flash-1017': V12_PROVIDERS,\n"
     "  'z-ai/glm-5.3-flash': V12_PROVIDERS,   // v1.3 board (D14); PIN recorded Oct 26 (Z.AI, fp8)\n", 1),
]


def main():
    old = SRC.read_text()
    text = old
    for a, b, n in EDITS:
        assert text.count(a) == n, (a[:80], text.count(a), n)
        text = text.replace(a, b)
    DST.write_text(text)
    diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                "sdl2-v1.2-20261026/arms/frontier/openrouter_transport.mjs",
                                "sdl2-v1.3-20261028/arms/frontier/openrouter_transport.mjs")
    (HERE / "openrouter_transport.diff").write_text("".join(diff))
    print(f"{len(EDITS)} edits -> {DST.name}")


if __name__ == "__main__":
    main()
