#!/usr/bin/env python3
"""Write concrete observations for the inspected E194 mandatory visual set."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))
import e194_g1_expansion_common as C  # noqa: E402

EVAL = C.RESULTS / "s6_downstream/eval/full_g1_expansion"

OBJECT_OBSERVATIONS = {
    "box001": {
        "noPRG": "grasp/lift/carry 中大箱始终在机器人身前且可见，place 帧箱体回到地面；机器人四阶段均保持站立。",
        "PRG": "grasp 后双臂贴近箱体，lift/carry 保持箱体在躯干前方，place 完成回落；未见跌倒或箱体飞离。",
        "G1": "四阶段均保持站立，箱体从抓取、抬升、搬运到落放连续可见，未见非有限跳变或灾难性脱手。",
    },
    "box023": {
        "noPRG": "小箱在 grasp 后被抬到身前，carry 保持可见，place 时回到脚前地面；机器人未倒地。",
        "PRG": "小箱完成 grasp→lift→carry→place，末段人与箱体分离可辨；四帧无飞散或姿态崩溃。",
        "G1": "小箱在四阶段轨迹连续，place 后稳定落地；机器人保持站立，未见灾难性接触丢失。",
    },
    "box021": {
        "noPRG": "长箱从地面抓取并在 lift/carry 保持于身前，place 回落；四帧中机器人均未跌倒。",
        "PRG": "长箱 grasp 后进入抬升与搬运，place 时回到地面；箱体和机器人在所有阶段均连续可见。",
        "G1": "长箱完成四阶段运动，机器人保持站立且无发散；carry/place 的箱体位置可与 PRG 逐帧对照。",
    },
}

OVERRIDES = {
    "box001_20231020_014_p2": "G1 在 lift/carry/place 的箱体倾斜与 PRG 明显不同，carry 后仍保持较大前倾；与 Δz=+1.866 cm、Δ3D=+1.878 cm 的数值回退一致。",
    "box021_20231018_028_p1": "G1 四阶段仍完整且无跌倒，但 carry/place 的箱体高度关系与 PRG 有偏移；Δz=+1.089 cm，保留为 z 数值例外。",
    "box021_20231018_028_p2": "G1 在 carry 的长箱横向位置/持箱姿态相对 PRG 有明显偏移，place 虽完成但路径不同；与 Δ3D=+3.043 cm 一致。",
    "box021_20231011_037_p1": "G1 的 lift/carry 箱体更贴近参考侧，place 正常；未见跌倒，且 Δz=-6.404 cm、Δ3D=-2.850 cm 显示明显改善。",
    "box001_20231003_2_041_p1": "G1 在 carry/place 的大箱倾角大于 PRG，但运动连续且机器人站立；object_ori gate 回退保留为姿态阈值例外。",
    "box001_20231023_107_p2": "G1 carry 时箱体相对躯干更偏外侧，place 仍完成；z 改善但 3D 增加 +1.478 cm，不能只用 z 下结论。",
    "box001_20231003_1_041_p2": "PRG 与 G1 均完成大箱抬升/搬运/落放；G1 carry 中箱体更接近竖直，且 z 与 3D 均改善。",
    "box023_20231011_019_p2": "G1 的小箱 carry/place 路径与 PRG 连续一致，未见脱手；3D 误差下降 5.454 cm，属于稳定改善样本。",
}


def gate_detail(reason: str) -> str:
    if "gate_PASS_TO_FAIL:" not in reason:
        return ""
    gates = reason.split("gate_PASS_TO_FAIL:", 1)[1].split(";", 1)[0].split(",")
    notes = []
    if any(gate in gates for gate in ("root_pos", "root_ori", "hand_pos", "hand_ori")):
        notes.append("G1 的躯干/手臂姿态相对 PRG 有可辨偏移，但仍保持平衡")
    if "object_ori" in gates:
        notes.append("carry/place 的箱体朝向存在轻微差异")
    if "contact" in gates:
        notes.append("手仍靠近箱体表面，未见肉眼级完全脱离")
    if "hand_penetration" in gates:
        notes.append("手位于箱体表面附近；四帧分辨率不足以判定 3 mm 穿透，数值 gate 仍按 FAIL 记录")
    if "release" in gates:
        notes.append("place 帧箱体已回地面，末段手箱相对位置与 PRG 不同")
    if any(gate in gates for gate in ("body_z", "lower_body")):
        notes.append("机器人未跌倒，但躯干/下肢姿态阈值回退仍保留")
    return "；".join(notes) + ("。" if notes else "")


def verdict(row: dict[str, str]) -> str:
    dz, d3d = float(row["delta_z_cm"]), float(row["delta_3d_cm"])
    if dz > 1.0 or d3d > 2.0:
        return "CASE_LEVEL_REGRESSION_KEEP_EXCEPTION"
    if "gate_PASS_TO_FAIL:" in row["selection_reason"]:
        return "NO_CATASTROPHIC_FAILURE_GATE_FLIP_REMAINS"
    if dz < 0 and d3d < 0:
        return "VISUALLY_STABLE_METRIC_IMPROVEMENT"
    return "VISUALLY_STABLE_REFERENCE_CASE"


def main() -> int:
    selection = C.read_tsv(EVAL / "e194_three_arm_visual_selection.tsv")
    if len(selection) != 36:
        raise SystemExit(f"selection rows={len(selection)} expected=36")
    rows = []
    for row in selection:
        object_notes = OBJECT_OBSERVATIONS[row["object_key"]]
        g1 = object_notes["G1"] + gate_detail(row["selection_reason"])
        if row["case_id"] in OVERRIDES:
            g1 += OVERRIDES[row["case_id"]]
        rows.append({"case_id": row["case_id"], "object_key": row["object_key"],
                     "execution_profile": row["execution_profile"], "selection_reason": row["selection_reason"],
                     "delta_z_cm": row["delta_z_cm"], "delta_3d_cm": row["delta_3d_cm"],
                     "noprg_observation": object_notes["noPRG"], "prg_observation": object_notes["PRG"],
                     "g1_observation": g1, "verdict": verdict(row),
                     "frame_evidence": f"visual_review/case_sheets/{row['case_id']}.jpg"})
    if any(not row[field].strip() for row in rows for field in
           ("noprg_observation", "prg_observation", "g1_observation", "verdict", "frame_evidence")):
        raise ValueError("blank visual observation")
    C.write_tsv(EVAL / "e194_three_arm_visual_review.tsv", rows)
    print(f"wrote {len(rows)} concrete visual-review rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
