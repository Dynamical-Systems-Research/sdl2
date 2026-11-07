"""Native Dynamical adapter; source-owned batch simulator, never physical execution.

This file is copied into an isolated, pinned runtime by prepare.py. It registers
through Dynamical's existing ABI and cannot admit itself. Full specimen records
are carried in a content-addressed sidecar, not reduced to aggregate science.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dynamical.instruments import InstrumentRequest, InstrumentResult, register
from dynamical.reasons import RuntimeReason

DEPENDENCY_LOCK_SHA256 = "5d136eba3ec5fc41236e907c97776d1a35dc069750144b4563366a8ec76bbe2f"
BUNDLE = Path(__file__).resolve().parents[1] / "bundle" / "ndfeb-sps"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked(path: Path, digest: str) -> object:
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or path.is_symlink():
        raise ValueError("invalid content-addressed artifact")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("artifact digest mismatch")
    return json.loads(data)


def exchange_root() -> Path:
    root = Path(os.environ["NDFEB_SPS_EXCHANGE_ROOT"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("owner exchange root must be an absolute non-symlink directory")
    return root.resolve(strict=True)


def verify_source() -> None:
    import ndfeb_sps

    root = Path(ndfeb_sps.__file__).resolve().parent
    lock = checked(BUNDLE / "dependency_lock.json", DEPENDENCY_LOCK_SHA256)
    for entry in lock["dependencies"]:
        path = root / entry["path"]
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("source dependency escapes package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("source dependency changed; independent re-review required")


@register("process-sps-batch", "ndfeb-sps-simulator")
def process_sps_batch(request: InstrumentRequest) -> InstrumentResult:
    start = time.perf_counter()
    try:
        if request.parameters or set(request.inputs) != {"owner_context_sha256"}:
            raise ValueError("batch request must use the declared owner context input")
        verify_source()
        root = exchange_root()
        digest = request.inputs["owner_context_sha256"]
        # Validate digest before it can participate in a path.
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid context digest")
        context = checked(root / f"{digest}.json", digest)
        if set(context) != {"request", "world_config", "seed"}:
            raise ValueError("context requires exactly request, world_config and seed")
        if context["request"].get("origin") != "synthetic":
            raise ValueError("physical requests cannot enter a simulator")
        world = context["world_config"]
        if not isinstance(world, dict):
            raise TypeError("world_config must declare an owner-side world object")
        family = world.get("family")
        if family == "rpi_conditional_workflow_v1":
            from ndfeb_sps.models.workflow_response import simulate_workflow

            delivery = simulate_workflow(
                context["request"], world_config=world, seed=context["seed"]
            )
        elif family is None:
            from ndfeb_sps.models import simulate_batch

            delivery = simulate_batch(
                context["request"], world_config=world, seed=context["seed"]
            )
        else:
            raise ValueError(f"unsupported conditional world family: {family!r}")
        if not isinstance(delivery, dict) or delivery.get("origin") != "synthetic":
            raise ValueError("response must retain synthetic origin")
        data = canonical(delivery)
        response_digest = hashlib.sha256(data).hexdigest()
        path = root / f"{response_digest}.json"
        try:
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o600)
        except FileExistsError:
            checked(path, response_digest)
        return InstrumentResult(
            outputs={
                "response_sha256": response_digest,
                "specimen_count": len(context["request"]["specimens"]),
            },
            uncertainty={},
            cost_usd=0.0,
            duration_s=time.perf_counter() - start,
            applied_parameters={},
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return InstrumentResult(
            outputs={"response_sha256": None, "specimen_count": None},
            uncertainty={},
            cost_usd=0.0,
            duration_s=time.perf_counter() - start,
            applied_parameters={name: None for name in request.parameters},
            reasons=[
                RuntimeReason(code="SPS_RESPONSE_UNAVAILABLE", detail=str(exc), recoverable=True)
            ],
        )
