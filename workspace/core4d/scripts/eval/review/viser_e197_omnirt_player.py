#!/usr/bin/env python3
"""Read-only Viser player for E197 OmniRetarget trajectories.

The player consumes the already-audited E197 converted scene_act qpos files,
not PRG rollout outputs.  It is intentionally separate from the multi-E170--
E194 review player: E197's absolute wide gate is an Omni-only candidate filter.
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
for path in (str(REPO), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from viser_review_player import (  # noqa: E402
    _build_scene,
    _compute_xforms,
    _load_portable_spec,
    case_name_matches,
)

OUT = REPO / "workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics"
METHOD_TSV = OUT / "e197_method_metrics.tsv"
GATE_TSV = OUT / "e197_omni_absolute_wide_gate_filter.tsv"
GATE_VERSION = "E197-omni-absolute-wide-v4"
GATES = (
    ("contact_3mm_in_mask", "3mm in-mask 接触", "≥", 0.01, "%"),
    ("raw_contact_in_mask", "Raw in-mask 接触", "≥", 0.50, "%"),
    ("hand_object_penetration_3mm", "手物 >3mm 穿透", "≤", 0.80, "%"),
    ("lower_body_penetration", "Lower-body 穿透", "≤", 0.30, "%"),
    ("foot_slip_max_m", "Foot slip max", "≤", 1.90, "m"),
    ("ankle_jerk_p95", "Ankle jerk P95", "≤", 4000.0, "m/s³"),
)
DISPLAY_METRICS = GATES + (("obj_speed_max", "Object speed max", "—", None, "m/s"),)


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    candidate = REPO / path
    if candidate.is_file():
        return candidate.resolve()
    for marker in ("workspace/", "example_datasets/"):
        if marker in value:
            candidate = REPO / (marker + value.split(marker, 1)[1])
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(value)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


@dataclass
class OmniRecord:
    case_id: str
    object_key: str
    source_exp: str
    retarget_variant_id: str
    qpos_path: Path
    scene_xml: Path
    frames: int
    metrics: dict[str, float]

    @property
    def failures(self) -> list[str]:
        failed = []
        for key, label, relation, threshold, _unit in GATES:
            if key == "contact_3mm_in_mask" and self.object_key == "box024":
                threshold = 0.0
            value = self.metrics[key]
            if (relation == "≥" and value < threshold) or (relation == "≤" and value > threshold):
                failed.append(key)
        return failed

    @property
    def gate_pass(self) -> bool:
        return not self.failures

    @property
    def playable(self) -> bool:
        return self.qpos_path.is_file() and self.scene_xml.is_file()


def load_records() -> list[OmniRecord]:
    if not METHOD_TSV.is_file():
        raise FileNotFoundError(METHOD_TSV)
    records = []
    with METHOD_TSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("method") != "OmniRetarget":
                continue
            metrics = {key: as_float(row, key) for key, *_ in DISPLAY_METRICS}
            records.append(
                OmniRecord(
                    case_id=row["case_id"],
                    object_key=row["object_key"],
                    source_exp=row["source_exp"],
                    retarget_variant_id=row["retarget_variant_id"],
                    qpos_path=repo_path(row["qpos_path"]),
                    scene_xml=repo_path(row["scene_xml"]),
                    frames=int(row["qpos_frames"]),
                    metrics=metrics,
                )
            )
    records.sort(key=lambda item: (item.object_key, item.case_id))
    if len(records) != 87 or len({item.case_id for item in records}) != 87:
        raise ValueError(f"expected 87 unique OmniRetarget rows, got {len(records)}")
    return records


def check(records: list[OmniRecord]) -> int:
    by_object = {key: sum(item.object_key == key for item in records) for key in sorted({item.object_key for item in records})}
    passed = [item for item in records if item.gate_pass]
    missing = [item.case_id for item in records if not item.playable]
    print(f"gate_version={GATE_VERSION}")
    print(f"records={len(records)} objects={by_object} playable={len(records) - len(missing)}/{len(records)}")
    print(f"omni_wide_pass={len(passed)}/{len(records)}")
    print("pass_by_object=" + str({key: sum(item.object_key == key for item in passed) for key in by_object}))
    if missing:
        print("missing=" + ",".join(missing))
    if GATE_TSV.is_file():
        with GATE_TSV.open(encoding="utf-8", newline="") as handle:
            gates = list(csv.DictReader(handle, delimiter="\t"))
        decisions = sum(row.get("rl_filter_decision") == "RL_CANDIDATE_OMNI_WIDE_GATE_PASS" for row in gates)
        if decisions != len(passed):
            print(f"filter mismatch: tsv={decisions}, recomputed={len(passed)}")
            return 1
    return 0 if not missing else 1


@functools.lru_cache(maxsize=None)
def case_frames(qpos_path: str, scene_xml: str):
    import mujoco

    spec = _load_portable_spec(Path(scene_xml))
    from spider.viewers.viser_viewer import _ensure_names

    _ensure_names(spec)
    model = spec.compile()
    qpos = np.asarray(np.load(qpos_path, allow_pickle=False)["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos {qpos.shape} incompatible with model nq={model.nq}")
    return spec, model, qpos


class OmniViserApp:
    def __init__(self, records: list[OmniRecord], host: str, port: int):
        import viser

        self.records = records
        self.server = viser.ViserServer(host=host, port=port)
        self.lock = threading.RLock()
        self.frames = []
        self.bodies = []
        self.visual_handles, self.collision_handles = [], []
        self.playing, self.next_time, self.programmatic = False, 0.0, False
        self.current: OmniRecord | None = None
        self._build_gui()
        self.refresh(initial=True)
        threading.Thread(target=self.player_loop, daemon=True).start()

    def _build_gui(self):
        with self.server.gui.add_folder("E197 Omni absolute wide gate"):
            self.metrics_md = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("筛选"):
            self.f_case = self.server.gui.add_text("Case 名称（支持子串）", initial_value="")
            self.f_object = self.server.gui.add_dropdown("物体", options=["全部"] + sorted({r.object_key for r in self.records}), initial_value="全部")
            self.f_gate = self.server.gui.add_dropdown("宽 gate", options=["全部", "通过", "过滤"], initial_value="全部")
            for handle in (self.f_case, self.f_object, self.f_gate):
                handle.on_update(lambda _=None: self.refresh())
        with self.server.gui.add_folder("样本"):
            self.case_dd = self.server.gui.add_dropdown("选择", options=["(无)"], initial_value="(无)")
            self.case_dd.on_update(self.on_pick)
            self.info_md = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("播放"):
            self.slider = self.server.gui.add_slider("帧", min=0, max=1, step=1, initial_value=0)
            self.slider.on_update(self.on_slider)
            self.server.gui.add_button("播放 / 暂停").on_click(self.toggle)
            self.fps = self.server.gui.add_number("播放帧率", initial_value=30, min=1, max=120, step=1)
        with self.server.gui.add_folder("显示"):
            self.show_collision = self.server.gui.add_checkbox("显示碰撞体", initial_value=False)
            self.show_grid = self.server.gui.add_checkbox("显示网格", initial_value=True)
            self.show_collision.on_update(lambda _: self.apply_visibility())
            self.show_grid.on_update(lambda _: self.reload_current())

    def filtered(self) -> list[OmniRecord]:
        out = []
        for record in self.records:
            if not case_name_matches(record.case_id, self.f_case.value):
                continue
            if self.f_object.value != "全部" and record.object_key != self.f_object.value:
                continue
            if self.f_gate.value == "通过" and not record.gate_pass:
                continue
            if self.f_gate.value == "过滤" and record.gate_pass:
                continue
            out.append(record)
        return out

    def label(self, record: OmniRecord) -> str:
        return f"{'✓' if record.gate_pass else '✗'} {record.object_key} {record.case_id} [{record.retarget_variant_id}]"

    def refresh(self, initial: bool = False):
        self.filtered_records = self.filtered()
        self.labels = [self.label(record) for record in self.filtered_records]
        options = self.labels or ["(无)"]
        self.case_dd.options = options
        self.case_dd.value = options[0]
        if self.filtered_records:
            self.load_case(self.filtered_records[0])
        elif not initial:
            self.info_md.content = "_当前筛选无匹配样本_"

    def on_pick(self, _=None):
        if self.case_dd.value in self.labels:
            self.load_case(self.filtered_records[self.labels.index(self.case_dd.value)])

    def reload_current(self):
        if self.current is not None:
            self.load_case(self.current)

    def load_case(self, record: OmniRecord):
        with self.lock:
            self.playing = False
            self.current = record
            self.server.scene.reset()
            self.frames, self.bodies = [], []
            self.visual_handles, self.collision_handles = [], []
            if not record.playable:
                self.info_md.content = f"### {record.case_id}\n\n⚠ qpos 或 scene XML 不存在"
                return
            try:
                spec, model, qpos = case_frames(str(record.qpos_path), str(record.scene_xml))
                if self.show_grid.value:
                    self.server.scene.add_grid("/grid")
                self.bodies, self.visual_handles, self.collision_handles = _build_scene(self.server, spec, model, "/omnirt")
                self.frames = _compute_xforms(model, qpos, [body_id for _, body_id in self.bodies], list(range(len(qpos))))
                self.slider.max = max(1, len(self.frames) - 1)
                self.set_slider(0)
                self.apply(0)
                self.apply_visibility()
                self.info_md.content = self.info(record)
                self.metrics_md.content = self.metric_table(record)
            except Exception as exc:
                self.info_md.content = f"### {record.case_id}\n\n⚠ 加载失败: `{exc}`"

    def metric_table(self, record: OmniRecord) -> str:
        heads, values = [], []
        for key, label, relation, threshold, unit in DISPLAY_METRICS:
            if key == "contact_3mm_in_mask" and record.object_key == "box024":
                threshold = 0.0
            heads.append(f"{label}{relation}{threshold:g}{unit}" if threshold is not None else label)
            value = record.metrics[key]
            text = f"{value:.1%}" if unit == "%" else f"{value:.1f} {unit}"
            failed = key in record.failures
            values.append(f"🔴**{text}**" if failed else text)
        return "| " + " | ".join(heads) + " |\n|" + "|".join(["---"] * len(heads)) + "|\n| " + " | ".join(values) + " |"

    def info(self, record: OmniRecord) -> str:
        failures = ", ".join(record.failures) or "—"
        decision = "✅ RL_CANDIDATE_OMNI_WIDE_GATE_PASS" if record.gate_pass else "❌ RL_CANDIDATE_FILTERED_OMNI_WIDE_GATE"
        return "\n".join((
            f"### OmniRetarget · {record.case_id}",
            f"- 宽 gate: {decision}",
            f"- 未通过项: {failures}",
            f"- 物体: **{record.object_key}** · source: **{record.source_exp}** · variant: **{record.retarget_variant_id}**",
            f"- qpos: `{record.frames}` frames · PRG 不参与该 gate",
        ))

    def player_loop(self):
        while True:
            if not self.playing or len(self.frames) < 2:
                time.sleep(0.03)
                continue
            now = time.perf_counter()
            if now >= self.next_time:
                frame = (int(self.slider.value) + 1) % len(self.frames)
                self.apply(frame)
                self.set_slider(frame)
                self.next_time = now + 1.0 / max(1, int(self.fps.value))
            else:
                time.sleep(0.003)

    def apply(self, frame: int):
        with self.lock, self.server.atomic():
            if not 0 <= frame < len(self.frames):
                return
            for handle, body_id in self.bodies:
                position, quat = self.frames[frame][body_id]
                handle.position = tuple(float(v) for v in position)
                handle.wxyz = tuple(float(v) for v in quat)

    def set_slider(self, frame: int):
        self.programmatic = True
        try:
            self.slider.value = int(frame)
        finally:
            self.programmatic = False

    def on_slider(self, _=None):
        if not self.programmatic:
            self.playing = False
            self.apply(int(self.slider.value))

    def toggle(self, _=None):
        self.playing = not self.playing
        self.next_time = time.perf_counter()

    def apply_visibility(self):
        for handle in self.visual_handles:
            handle.visible = True
        for handle in self.collision_handles:
            handle.visible = bool(self.show_collision.value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--check", action="store_true", help="audit all E197 Omni replay assets and exit")
    args = parser.parse_args()
    records = load_records()
    if args.check:
        return check(records)
    OmniViserApp(records, args.host, args.port)
    print(f"[E197 Omni] Viser server: http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
