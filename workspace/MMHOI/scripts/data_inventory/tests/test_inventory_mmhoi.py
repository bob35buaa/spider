from __future__ import annotations

import json
import sys
import unittest
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from inventory_mmhoi import (  # noqa: E402
    COLLABORATIVE_SCENARIOS,
    PRIMARY_SCENARIOS,
    ActionRow,
    assign_split,
    inventory_archive,
    normalize_sequence_key,
    parse_action_rows,
    parse_scenario_code,
)


class InventoryUnitTests(unittest.TestCase):
    """Unit coverage for parsing and split helpers."""

    def test_parse_scenario_code(self) -> None:
        """Scenario folder variants map to canonical codes."""
        cases = {
            "20240412__C_2__30skip": "C_2",
            "20240508_C_9__30skip": "C_9",
            "20240508_C_9__3person_after_30skip": "C_9_r2",
            "20240510_C_1_3person__30skip": "C_1_r2",
            "20240412_C_5K3__30skip": "C_5",
        }
        for folder_name, expected in cases.items():
            with self.subTest(folder_name=folder_name):
                self.assertEqual(parse_scenario_code(folder_name), expected)

    def test_collaborative_scenario_mapping_is_explicit(self) -> None:
        """Collaborative and primary scenario scopes stay explicit."""
        self.assertEqual(
            set(COLLABORATIVE_SCENARIOS),
            {"C_2", "C_8", "C_9", "C_10", "C_9_r2"},
        )
        self.assertEqual(
            PRIMARY_SCENARIOS,
            {
                "C_2": "Moving heavy stuffs 1",
                "C_8": "Moving heavy stuffs 2",
            },
        )

    def test_normalize_sequence_key_handles_single_person_alias(self) -> None:
        """The known sequence alias maps to the official split key."""
        self.assertEqual(
            normalize_sequence_key("Single20240531_personA_all_30skip_start-end"),
            "Single20240531_single_personA_all_30skip_start-end",
        )
        self.assertEqual(
            normalize_sequence_key("20240412_personA_personB"),
            "20240412_personA_personB",
        )

    def test_assign_split(self) -> None:
        """Chronological indices map to declared split counts."""
        cases = {
            0: "train",
            1: "train",
            2: "val",
            3: "test",
        }
        for index, expected in cases.items():
            with self.subTest(index=index):
                self.assertEqual(assign_split(index, (2, 1, 1)), expected)

    def test_assign_split_rejects_out_of_range_index(self) -> None:
        """Split assignment rejects indices beyond declared totals."""
        with self.assertRaisesRegex(ValueError, "outside split counts"):
            assign_split(4, (2, 1, 1))

    def test_parse_action_rows_parses_person_object_and_verb(self) -> None:
        """Action CSV parsing keeps the semantic tail columns."""
        rows = parse_action_rows(
            "1,2,3,4,5,6,7,8,person1,table_wood,move together\n"
            "1,2,3,4,5,6,7,8,person2,table_wood,move together\n"
        )
        self.assertEqual(
            rows,
            (
                ActionRow(
                    person="person1",
                    object_name="table_wood",
                    verb="move together",
                ),
                ActionRow(
                    person="person2",
                    object_name="table_wood",
                    verb="move together",
                ),
            ),
        )

    def test_parse_action_rows_rejects_malformed_row(self) -> None:
        """Malformed action rows fail with an explicit column error."""
        with self.assertRaisesRegex(ValueError, "expected at least 11 columns"):
            parse_action_rows("person1,table_wood,move together\n")


def _write_action_csv(
    archive: zipfile.ZipFile,
    sequence: str,
    scenario_folder: str,
    frame: str,
    rows: tuple[tuple[str, str, str], ...],
) -> None:
    csv_rows = [
        ("1", "2", "3", "4", "5", "6", "7", "8", person, obj, verb)
        for person, obj, verb in rows
    ]
    text_lines: list[str] = []
    for row in csv_rows:
        output = []
        output.extend(row)
        text_lines.append(",".join(output))
    path = (
        f"MMHOI/sequences/{sequence}/{scenario_folder}/{frame}/PARAM/action.csv"
    )
    archive.writestr(path, "\n".join(text_lines) + "\n")


class InventoryIntegrationTests(unittest.TestCase):
    """Small-ZIP integration coverage for inventory policies."""

    def test_inventory_archive_counts_collaborative_samples(self) -> None:
        """The mini archive yields expected collaborative scopes."""
        with self.subTest("mini archive"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "mini_mmhoi.zip"
                self._write_mini_archive(archive_path)
                result = inventory_archive(archive_path)

        self.assertEqual(result["archive"]["entry_count"], 7)
        self.assertEqual(result["global"]["object_mesh_count"], 2)
        self.assertEqual(result["collaborative"]["scenario_capture_count"], 2)
        self.assertEqual(result["collaborative"]["annotated_sample_count"], 3)
        self.assertEqual(
            result["collaborative"]["split_sample_counts"],
            {"train": 2, "val": 0, "test": 1, "unspecified": 0},
        )
        self.assertEqual(result["split_audit"]["mismatch_count"], 0)
        self.assertEqual(result["collaborative"]["action_row_count"], 7)
        self.assertEqual(
            result["collaborative"]["active_object_types"],
            ["box", "table_wood"],
        )
        self.assertEqual(result["collaborative"]["cooperative_sample_count"], 1)
        self.assertEqual(
            result["collaborative"]["consecutive_frame_gap_distribution"],
            {"30": 1},
        )
        self.assertEqual(
            result["collaborative"]["strict_two_person"][
                "annotated_sample_count"
            ],
            2,
        )
        self.assertEqual(
            result["collaborative"]["strict_two_person"][
                "split_sample_counts"
            ],
            {"train": 1, "val": 0, "test": 1, "unspecified": 0},
        )
        self.assertEqual(
            result["primary_scope"]["scenario_codes"],
            {"C_2": "Moving heavy stuffs 1"},
        )
        self.assertEqual(result["primary_scope"]["annotated_sample_count"], 2)
        self.assertEqual(
            result["primary_scope"]["consecutive_frame_gap_distribution"],
            {"30": 1},
        )
        self.assertEqual(
            result["pilot_c2_box"]["selection"]["annotated_sample_count"],
            1,
        )
        self.assertEqual(
            result["pilot_c2_box"]["selection"]["split_sample_counts"],
            {"train": 1, "val": 0, "test": 0, "unspecified": 0},
        )
        self.assertEqual(
            result["pilot_c2_box"]["target_object_action_row_count"],
            4,
        )
        self.assertEqual(
            result["pilot_c2_box"]["target_object_active_action_row_count"],
            2,
        )
        self.assertEqual(
            result["pilot_c2_box"]["target_object_cooperative_sample_count"],
            1,
        )
        self.assertEqual(
            result["primary_temporal_audit"]["numeric_frame_folder_count"],
            2,
        )
        self.assertTrue(
            result["primary_temporal_audit"][
                "all_capture_folders_marked_30skip"
            ]
        )
        self.assertEqual(
            result["primary_temporal_audit"]["dense_candidate_file_count"],
            0,
        )

    def test_inventory_archive_preserves_samples_beyond_split_counts(self) -> None:
        """Archive samples beyond split totals become unspecified."""
        with self.subTest("split overflow"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "mini_mmhoi_overflow.zip"
                self._write_mini_archive(archive_path, c2_counts=[1, 0, 0])
                result = inventory_archive(archive_path)

        self.assertEqual(
            result["collaborative"]["split_sample_counts"],
            {"train": 2, "val": 0, "test": 0, "unspecified": 1},
        )
        self.assertEqual(result["split_audit"]["mismatch_count"], 1)

    def test_inventory_archive_preserves_samples_with_missing_split_key(self) -> None:
        """Samples without a split key remain as unspecified evidence."""
        with self.subTest("missing split key"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "mini_mmhoi_missing_split.zip"
                self._write_mini_archive(archive_path, include_c2_split=False)
                result = inventory_archive(archive_path)

        self.assertEqual(
            result["collaborative"]["split_sample_counts"],
            {"train": 1, "val": 0, "test": 0, "unspecified": 2},
        )
        self.assertEqual(result["split_audit"]["mismatch_count"], 1)
        mismatch = result["tables"]["split_mismatches"][0]
        self.assertEqual(mismatch["reason"], "missing_split_key")

    @staticmethod
    def _write_mini_archive(
        archive_path: Path,
        c2_counts: list[int] | None = None,
        include_c2_split: bool = True,
    ) -> None:
        split_rows = {
            "C_5": [1, 0, 0],
            "C_9_r2": [1, 0, 0],
        }
        if include_c2_split:
            split_rows["C_2"] = c2_counts or [1, 0, 1]
        split_data = {
            "20240412_personA_personB": split_rows
        }
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "MMHOI/splits/train_val_test_split.json",
                json.dumps(split_data),
            )
            archive.writestr("MMHOI/object/01_bat.ply", "ply\n")
            archive.writestr("MMHOI/object/12_table_wood.ply", "ply\n")
            _write_action_csv(
                archive,
                "20240412_personA_personB",
                "20240412_C_2__30skip",
                "00001",
                (
                    ("person1", "box", "move together"),
                    ("person2", "box", "move together"),
                ),
            )
            _write_action_csv(
                archive,
                "20240412_personA_personB",
                "20240412_C_2__30skip",
                "00031",
                (
                    ("person1", "box", "no-interaction"),
                    ("person2", "box", "no-interaction"),
                ),
            )
            _write_action_csv(
                archive,
                "20240412_personA_personB",
                "20240412_C_5__30skip",
                "00001",
                (("person1", "mug", "hold"),),
            )
            _write_action_csv(
                archive,
                "20240412_personA_personB",
                "20240412_C_9__3person_after_30skip",
                "00001",
                (
                    ("person1", "table_wood", "organize"),
                    ("person2", "table_wood", "organize"),
                    ("person3", "table_wood", "organize"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
