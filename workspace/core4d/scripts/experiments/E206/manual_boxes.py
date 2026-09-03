#!/usr/bin/env python3
"""E206 P2.3b: hand-authored collision boxes, stored as absolute geometry.

`box_edits.json` records reviewer intent as ORIGINAL BUILD-ORDER INDICES, which
is only meaningful against the exact auto build the reviewer was looking at.
That coupling has already produced one corrupt record (chair020, 2026-09-03: a
click landed on a 6-box semantic proxy while the file still held indices from a
15-box voxel proxy, leaving `removed=[12,13,14]` against `n_boxes_original=6`).

The auto proxies also cannot express what the reviewer actually wants for the
worst objects: chair005 and chair022 sit at 41% cavity over-fill because a
greedy voxel merge or a 4-way leg split cannot leave the under-seat gap open at
any budget.  Deleting boxes does not fix that; re-placing them does.

So this module stores the boxes THEMSELVES — centre + half-size in the object
body frame, exactly the pair `geom_box_xml` emits.  No indices, no fingerprint,
nothing to go stale: a manual record is self-describing and survives any change
to the auto path.  When present it wins over voxel/semantic (see
`semantic_proxy.build_effective_proxy`), which are demoted to seeds.

Axis-aligned only.  `spider/config.py` fail-closes the union SDF on non-box
geoms, and E206 deliberately does not introduce `quat` on object collision
geoms this round.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
from lowgeom_proxy import ProxyBox  # noqa: E402

MANUAL_PATH = C.S2_PROXY_DIR / "manual_boxes.json"

# A box thinner than this is almost certainly a mis-drag, not an intent.
MIN_HALF_M = 0.004


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_manual(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else MANUAL_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manual(records: dict[str, Any], path: Path | None = None) -> None:
    path = Path(path) if path is not None else MANUAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(boxes: list[ProxyBox], object_key: str, *, n_max: int | None = None) -> None:
    """Reject a record that could not possibly compile into a scene.

    The box-count ceiling is only a warning here: G1 in
    `audit_lowgeom_contract.py` is the authority on the budget, and blocking a
    save mid-edit would just lose the reviewer's work.
    """
    if not boxes:
        raise ValueError(f"{object_key}: a manual proxy needs at least one box")
    for index, box in enumerate(boxes):
        center = np.asarray(box.center, dtype=np.float64)
        half = np.asarray(box.half_size, dtype=np.float64)
        if center.shape != (3,) or half.shape != (3,):
            raise ValueError(f"{object_key} box {index}: expected 3-vectors")
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(half))):
            raise ValueError(f"{object_key} box {index}: non-finite values")
        if np.any(half < MIN_HALF_M):
            raise ValueError(
                f"{object_key} box {index}: half_size {half} below {MIN_HALF_M} m"
            )
    if n_max is not None and len(boxes) > n_max:
        print(
            f"[E206] warning: {object_key} has {len(boxes)} manual boxes > n_max={n_max}; "
            "audit G1 will fail until you merge or delete some",
            flush=True,
        )


def to_proxy_boxes(record: dict[str, Any]) -> list[ProxyBox]:
    return [
        ProxyBox(
            center=np.asarray(b["center"], dtype=np.float64),
            half_size=np.asarray(b["half_size"], dtype=np.float64),
        )
        for b in record["boxes"]
    ]


def manual_boxes_for(
    object_key: str, *, records: dict[str, Any] | None = None
) -> tuple[list[ProxyBox], list[str]] | None:
    """The hand-authored proxy for this object, or None if there isn't one."""
    records = load_manual() if records is None else records
    record = records.get(object_key)
    if not record or not record.get("boxes"):
        return None
    boxes = to_proxy_boxes(record)
    labels = [str(b.get("label") or f"m{i:02d}") for i, b in enumerate(record["boxes"])]
    return boxes, labels


def record_manual(
    object_key: str,
    boxes: list[ProxyBox],
    labels: list[str],
    *,
    seed: dict[str, Any] | None = None,
    editor: str = "user",
    notes: str = "",
    n_max: int | None = None,
    records: dict[str, Any] | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Write (or overwrite) the manual proxy for one object."""
    validate(boxes, object_key, n_max=n_max)
    records = load_manual() if records is None else records
    records[object_key] = {
        "boxes": [
            {
                "center": [float(v) for v in np.asarray(box.center)],
                "half_size": [float(v) for v in np.asarray(box.half_size)],
                "label": labels[i] if i < len(labels) else f"m{i:02d}",
            }
            for i, box in enumerate(boxes)
        ],
        "seed": seed or {},
        "editor": editor,
        "edited_at": now(),
        "notes": notes,
    }
    if save:
        save_manual(records)
    return records


def drop_manual(object_key: str, *, records: dict[str, Any] | None = None) -> dict[str, Any]:
    """Discard the manual proxy so the object falls back to the auto path."""
    records = load_manual() if records is None else records
    records.pop(object_key, None)
    save_manual(records)
    return records
