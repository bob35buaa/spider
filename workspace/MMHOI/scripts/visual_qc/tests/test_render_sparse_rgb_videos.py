from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from render_sparse_rgb_videos import (  # noqa: E402
    CaseSpec,
    discover_rgb_frames,
    frame_gap_distribution,
    load_case_specs,
    safe_case_slug,
)


class SparseRgbVideoUnitTests(unittest.TestCase):
    """Unit coverage for sparse RGB discovery and manifest parsing."""

    def test_discover_rgb_frames_orders_sparse_frame_ids(self) -> None:
        """Camera RGB discovery follows numeric source-frame order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            for frame_id in ("00061", "00001", "00031"):
                frame_dir = case_dir / frame_id
                frame_dir.mkdir()
                (frame_dir / f"0_{frame_id}.jpg").touch()
                (frame_dir / f"1_{frame_id}.jpg").touch()

            frames = discover_rgb_frames(case_dir, camera_id=0)

        self.assertEqual(
            [(frame.frame_id, frame.image_path.name) for frame in frames],
            [
                (1, "0_00001.jpg"),
                (31, "0_00031.jpg"),
                (61, "0_00061.jpg"),
            ],
        )

    def test_discover_rgb_frames_rejects_missing_camera_frame(self) -> None:
        """A case with no requested camera RGB fails explicitly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            (case_dir / "00001").mkdir()
            (case_dir / "00001" / "1_00001.jpg").touch()

            with self.assertRaisesRegex(ValueError, "no camera-0 RGB frames"):
                discover_rgb_frames(case_dir, camera_id=0)

    def test_frame_gap_distribution_counts_source_frame_gaps(self) -> None:
        """Frame-gap counts preserve nonstandard gaps."""
        self.assertEqual(
            frame_gap_distribution((1, 31, 61, 121)),
            {30: 2, 60: 1},
        )

    def test_safe_case_slug_is_stable_and_path_free(self) -> None:
        """Case slugs remove path separators and repeated punctuation."""
        self.assertEqual(
            safe_case_slug(
                "20240412_personA_personB/20240412__C_8__30skip"
            ),
            "20240412_personA_personB__20240412_C_8_30skip",
        )

    def test_load_case_specs_validates_required_columns(self) -> None:
        """A valid selected-case TSV maps to immutable case specs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "cases.tsv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "case_id",
                        "scenario_code",
                        "selection_reason",
                    ),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": (
                            "20240412_personA_personB/"
                            "20240412__C_8__30skip"
                        ),
                        "scenario_code": "C_8",
                        "selection_reason": "user example",
                    }
                )

            specs = load_case_specs(manifest)

        self.assertEqual(
            specs,
            (
                CaseSpec(
                    case_id=(
                        "20240412_personA_personB/"
                        "20240412__C_8__30skip"
                    ),
                    scenario_code="C_8",
                    selection_reason="user example",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
