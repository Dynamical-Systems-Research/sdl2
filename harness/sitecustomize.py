"""Route the frozen world package to the released leaderboard package. Installed by setup.sh in both environments.

World records name their package by absolute path and sha256, and both seed each world's random streams, so the records
stay unchanged. When ndfeb_sps.models.c7_campaign is imported, load_package is wrapped so that the frozen package record
loads the released package named in sdl2_package.json beside this file.
"""
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

_CFG = Path(__file__).with_name("sdl2_package.json")


def _patch(module):
    cfg = json.loads(_CFG.read_text())
    original = module.load_package
    released = {k: v for k, v in cfg["package_sha256"].items()}

    def load_package(package):
        if package.get("sha256") == cfg["frozen_sha256"]:
            return original({"dir": cfg["package_dir"], "sha256": released})
        return original(package)
    module.load_package = load_package


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "ndfeb_sps.models.c7_campaign":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)
        spec.loader.exec_module = exec_module
        return spec


if _CFG.exists():
    sys.meta_path.insert(0, _Finder())
