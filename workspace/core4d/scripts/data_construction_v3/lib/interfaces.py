#!/usr/bin/env python3
"""Stable extension interfaces for Core4D data-construction v3.

These protocols are intentionally lightweight. They define result contracts for
future filters/adapters/gates without forcing the current scripts into a plugin
framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any, Protocol

from common import SCHEMA_VERSION, json_dumps, timestamp


DECISION_VALUES = {"pass", "review", "reject", "not_run"}
STATUS_VALUES = {"pass", "review", "reject", "not_run", "fail", "missing", "error"}


def stable_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class ExtensionResult:
    """Base result row shared by all extension points."""

    interface_name: str
    case_id: str = ""
    decision: str = "not_run"
    status: str = "not_run"
    reason: str = ""
    metrics_json: str = "{}"
    evidence_paths_json: str = "{}"
    schema_version: str = SCHEMA_VERSION
    updated_at: str = field(default_factory=timestamp)

    def validate(self) -> None:
        if self.decision not in DECISION_VALUES:
            raise ValueError(f"{self.interface_name}: invalid decision={self.decision}")
        if self.status not in STATUS_VALUES:
            raise ValueError(f"{self.interface_name}: invalid status={self.status}")
        json.loads(self.metrics_json or "{}")
        json.loads(self.evidence_paths_json or "{}")

    def to_row(self) -> dict[str, str]:
        self.validate()
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass
class CandidateFilterResult(ExtensionResult):
    interface_name: str = "CandidateFilter"


@dataclass
class RetargetAdapterResult(ExtensionResult):
    interface_name: str = "RetargetAdapter"
    retarget_variant_id: str = ""
    target_variant_id: str = "ref_fk"
    converted_npz: str = ""
    omniretarget_output_npz: str = ""
    trimmed_npz: str = ""
    params_json: str = "{}"

    def validate(self) -> None:
        super().validate()
        json.loads(self.params_json or "{}")


@dataclass
class TemplateBuilderResult(ExtensionResult):
    interface_name: str = "TemplateBuilder"
    source_scene_task: str = ""
    scene_xml: str = ""
    task_info_json: str = ""
    template_adapter: str = ""
    proxy_template: str = "False"
    collision_policy: str = ""


@dataclass
class TargetGateResult(ExtensionResult):
    interface_name: str = "TargetGate"
    retarget_variant_id: str = ""
    target_variant_id: str = "ref_fk"
    target_scene: str = ""
    trajectory: str = ""
    failure_mode: str = ""


@dataclass
class VisualizerResult(ExtensionResult):
    interface_name: str = "Visualizer"
    video_path: str = ""
    sheet_path: str = ""
    manifest_path: str = ""


class CandidateFilter(Protocol):
    def evaluate(self, case_row: dict[str, str], context: dict[str, Any]) -> CandidateFilterResult:
        """Return a pass/review/reject decision for one case row."""


class RetargetAdapter(Protocol):
    def run(self, case_row: dict[str, str], variant_row: dict[str, str], out_dir: Path, dry_run: bool) -> RetargetAdapterResult:
        """Run or plan one retarget/preprocess adapter invocation."""


class TemplateBuilder(Protocol):
    def build_or_audit(self, template_row: dict[str, str], out_dir: Path, dry_run: bool) -> TemplateBuilderResult:
        """Build or audit one source template."""


class TargetGate(Protocol):
    def evaluate(self, target_row: dict[str, str], context: dict[str, Any]) -> TargetGateResult:
        """Evaluate whether one target can enter downstream optimization."""


class Visualizer(Protocol):
    def render(self, row: dict[str, str], out_dir: Path, overwrite: bool) -> VisualizerResult:
        """Render visual evidence and return manifest paths."""


def self_test() -> None:
    rows = [
        CandidateFilterResult(case_id="case", decision="pass", status="pass", metrics_json=stable_json({"score": 1.0})),
        RetargetAdapterResult(
            case_id="case",
            decision="pass",
            status="pass",
            retarget_variant_id="omnirt_v1",
            converted_npz="/tmp/converted.npz",
            omniretarget_output_npz="/tmp/retargeted.npz",
            trimmed_npz="/tmp/trimmed.npz",
            params_json=stable_json({"replace_wrist_with_fingertip": False}),
        ),
        TemplateBuilderResult(
            case_id="case",
            decision="review",
            status="review",
            source_scene_task="bucket007_person1",
            template_adapter="nonbox_proxy_aabb_review",
            proxy_template="True",
            collision_policy="bucket_wall_proxy_aabb",
        ),
        TargetGateResult(case_id="case", decision="reject", status="reject", failure_mode="target_gate_exception"),
        VisualizerResult(case_id="case", decision="pass", status="pass", video_path="/tmp/a.mp4", sheet_path="/tmp/a.png"),
    ]
    for row in rows:
        row.to_row()
    print(json_dumps({"status": "pass", "interfaces": [row.interface_name for row in rows]}))


if __name__ == "__main__":
    self_test()
