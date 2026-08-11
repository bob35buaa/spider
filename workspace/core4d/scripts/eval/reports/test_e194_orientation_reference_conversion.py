from __future__ import annotations

import csv
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
sys.path.insert(0, str(SCRIPT.parent))

from e194_orientation_reference_conversion import audit_case  # noqa: E402


EVAL_ROOT = REPO / "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion"
FOCAL_CASE = "box001_20231003_2_041_p1"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_focal_runtime_reference_uses_wrong_euler_order() -> None:
    metrics = read_tsv(EVAL_ROOT / "e194_three_arm_case_metrics.tsv")
    paired = read_tsv(EVAL_ROOT / "e194_three_arm_paired_deltas.tsv")
    metric = next(
        row for row in metrics if row["case_id"] == FOCAL_CASE and row["arm"] == "G1"
    )
    delta_row = next(
        row
        for row in paired
        if row["case_id"] == FOCAL_CASE and row["comparison"] == "PRG_to_G1"
    )
    log_path = REPO / f"logs/E194/cem/full_g1_expansion/{metric['variant']}.log"

    result = audit_case(
        metric,
        float(delta_row["delta_track_obj_ori_err_deg_mean"]),
        log_path,
    )

    assert result["runtime_euler_convention"] == "XYZ"
    assert result["xml_hinge_axis_sequence"] == "XZY"
    assert result["runtime_convention_matches_xml_axes"] is False
    assert result["runtime_target_vs_raw_ori_err_deg_mean"] > 25.0
    assert result["axis_target_vs_raw_ori_err_deg_max"] < 1e-4
    assert result["g1_vs_runtime_target_ori_err_deg_mean"] < 10.0
    assert result["g1_vs_raw_ori_err_reproduction_abs_diff_deg"] < 1e-6


if __name__ == "__main__":
    test_focal_runtime_reference_uses_wrong_euler_order()
    print("PASS: focal runtime reference conversion audit")
