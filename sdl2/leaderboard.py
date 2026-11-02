"""Route the frozen world package to the released leaderboard package.

World records name their package by absolute path and sha256, and both seed each world's random streams, so the records
are released unchanged. The released package (`worlds/board-package` in the dataset) keeps the frozen files and array
shapes, with every row outside the 32 leaderboard worlds set to NaN and the anneal refit's seed withheld.
"""
import hashlib
import json
import sys
from pathlib import Path

from . import FROZEN

FROZEN_PACKAGE_SHA256_FILE = "FROZEN_PACKAGE_SHA256.json"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_paths(root=None):
    """Put the frozen mining scripts, grader, study source, twin package and v1.2 worlds on sys.path."""
    root = Path(root or FROZEN)
    entries = [str(root / d) for d in ("mining", "grader", "src", "", "worlds_v12")]
    for p in reversed(entries):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return entries


def install(package_dir, frozen=None):
    """Make c7_campaign.load_package load the released package whenever a record names the frozen one. Idempotent."""
    frozen_paths(frozen)
    from ndfeb_sps.models import c7_campaign as c7
    if getattr(c7.load_package, "_sdl2_package", None) == str(package_dir):
        return
    package_dir = Path(package_dir).resolve()
    frozen_sha256 = json.loads((package_dir / FROZEN_PACKAGE_SHA256_FILE).read_text())
    released_sha256 = {k: _sha(package_dir / c7.PACKAGE_FILES[k]) for k in frozen_sha256 if k != "manifest"}
    original = getattr(c7.load_package, "_sdl2_original", c7.load_package)

    def load_package(package):
        if package["sha256"] == frozen_sha256:
            return original({"dir": str(package_dir), "sha256": released_sha256})
        return original(package)
    load_package._sdl2_package = str(package_dir)
    load_package._sdl2_original = original
    c7.load_package = load_package
