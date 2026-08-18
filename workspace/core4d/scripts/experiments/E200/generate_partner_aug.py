#!/usr/bin/env python3
"""Resolve (and optionally generate) augmented partner motions for the E200 RL export.

Each augmented (trans0/1/2) source case needs its bimanual partner (the opposite
CORE4D person in the same take) retargeted under the SAME per-person aug label.
About two thirds of partner motions already exist in the E199 holosoma retarget
tree and are reused; the rest are generated here by re-running the E199 upstream
(convert -> parallel_robot_retarget --augmentation omnirt_v2 -> fixed-window trim)
for the partner person -- deterministic, per-person human frame, same trans_k.

Partner metadata (date/seq/object/model) is taken from the source aug task_info
with the person flipped, so partners whose own base task_info is absent are still
handled uniformly. Produces ``partner_aug_motion_index.tsv`` consumed by
build_rl_export.py. This is downstream-only and does not touch S1-S5 or CEM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

E199_DIR = Path(__file__).resolve().parents[1] / "E199"
sys.path.insert(0, str(E199_DIR))
import build_augmented_tasks as B  # noqa: E402
import e199_common as C  # noqa: E402

PERSON_FLIP = {"person1": "person2", "person2": "person1"}
PERSON_SHORT = {"person1": "p1", "person2": "p2"}
PERSON_IDX = {"person1": "0", "person2": "1"}
SHORT_PERSON = {"p1": "person1", "p2": "person2"}
VARIANT_HOLO = {"trans0": "trans_0", "trans1": "trans_1", "trans2": "trans_2"}

# per-object arm -> priority manifest that carries its aug rows
ARM_MANIFEST = {
    "box001": C.RESULTS.parent / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box024": C.RESULTS.parent / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box004": C.RESULTS.parent / "E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv",
    "box023": C.RESULTS.parent / "E200/s6_downstream/manifests/e200_noprg_priority_manifest.tsv",
    "box021": C.RESULTS / "s6_downstream/manifests/e199_fullscale_priority_manifest.tsv",
}

INDEX_FIELDS = [
    "source_case_id",
    "source_variant",
    "object_key",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "partner_base_task",
    "partner_holosoma_task",
    "partner_trimmed_npz",
    "partner_omniretarget_output_npz",
    "partner_trim_window_json",
    "partner_status",
    "partner_generation_mode",
]


def norm_pn(case_id: str) -> str:
    """Normalize a case id to the ``_p1``/``_p2`` short-person form."""
    return case_id.replace("_person1", "_p1").replace("_person2", "_p2")


def flip_case(case_pn: str) -> tuple[str, str]:
    """Return (partner_case_id, partner_person) by flipping the person suffix."""
    if case_pn.endswith("_p1"):
        return case_pn[:-3] + "_p2", "person2"
    if case_pn.endswith("_p2"):
        return case_pn[:-3] + "_p1", "person1"
    raise SystemExit(f"unexpected case id (no _pN suffix): {case_pn}")


def load_priority_index() -> dict[tuple[str, str, str], dict[str, str]]:
    """Index aug rows by (object, case_id_pn, variant), each object from ITS arm's manifest.

    An object appears in multiple arms' manifests; take each object only from the
    manifest assigned to it to avoid cross-arm key collisions. (The aug SPIDER task
    is arm-independent, so partner resolution is unaffected either way, but this
    keeps the source-of-truth unambiguous.)
    """
    idx: dict[tuple[str, str, str], dict[str, str]] = {}
    for obj, path in ARM_MANIFEST.items():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["object_key"] != obj:
                    continue
                idx[(obj, norm_pn(row["case_id"]), row["aug_variant"])] = row
    return idx


def partner_meta(source_aug_task: str, partner_person: str) -> dict[str, str]:
    """Derive partner retarget metadata from the source aug task_info (flip person).

    ``source_scene_task`` MUST be non-empty: pipeline.sh reads the case file with
    ``IFS=$'\t'`` and tab is a whitespace IFS char, so an empty field collapses and
    shifts ``target_task`` (yielding ``holosoma_auto``). It is unused under
    --skip-spider, so the source's geometry-template task name is a safe filler.
    """
    info_path = C.TASK_ROOT / source_aug_task / "task_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    date, seq, obj = info["date"], info["seq"], info["object_name"]
    source_scene_task = Path(info.get("source_scene", "")).parent.name or source_aug_task
    return {
        "date": date,
        "seq": seq,
        "person": partner_person,
        "object_name": obj,
        "object_model_rel": info["object_model_rel"],
        "source_scene_task": source_scene_task,  # non-empty filler; unused under --skip-spider
        "holosoma_task": f"{date}-{seq}-{partner_person}-{obj}_with_obj",
    }


def resolve_partner(meta: dict[str, str], partner_base_task: str, variant: str):
    """Return (trimmed, retargeted, trim_window) paths if the aug motion exists."""
    root = B.case_root(partner_base_task)
    holo = meta["holosoma_task"]
    holo_variant = VARIANT_HOLO[variant]
    trimmed = root / "trimmed" / f"{holo}_{holo_variant}.npz"
    retargeted = root / "retargeted" / f"{holo}_{holo_variant}.npz"
    trim_window = root / "trim_window.json"
    if trimmed.is_file() and trimmed.stat().st_size > 0:
        return trimmed, retargeted, trim_window
    return None


def selection_aug_rows(selection_tsv: Path) -> list[dict[str, str]]:
    """Return the augmented rows of a selection.tsv."""
    with selection_tsv.open(encoding="utf-8", newline="") as stream:
        return [r for r in csv.DictReader(stream, delimiter="\t") if r["orig_or_aug"] == "aug"]


def build_index(
    aug_rows: list[dict[str, str]],
    priority: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    """Resolve the partner motion for every aug case (present vs missing)."""
    rows: list[dict[str, str]] = []
    for sel in aug_rows:
        obj, src_case, variant = sel["object"], norm_pn(sel["case_id"]), sel["variant"]
        man = priority.get((obj, src_case, variant))
        if man is None:
            raise SystemExit(f"aug case not in priority manifest: {obj} {src_case} {variant}")
        source_aug_task = man["target_task"]
        partner_case, partner_person = flip_case(src_case)
        partner_base_task = f"dcv3_omnirt_v1_ref_fk_{partner_case}"
        meta = partner_meta(source_aug_task, partner_person)
        found = resolve_partner(meta, partner_base_task, variant)
        row = {
            "source_case_id": src_case,
            "source_variant": variant,
            "object_key": obj,
            "partner_case_id": partner_case,
            "partner_person": partner_person,
            "partner_person_idx": PERSON_IDX[partner_person],
            "partner_base_task": partner_base_task,
            "partner_holosoma_task": meta["holosoma_task"],
            "partner_trimmed_npz": C.rel(found[0]) if found else "",
            "partner_omniretarget_output_npz": C.rel(found[1]) if found else "",
            "partner_trim_window_json": C.rel(found[2]) if found else "",
            "partner_status": "present" if found else "missing",
            "partner_generation_mode": "reuse_e199_partner_aug" if found else "",
        }
        rows.append(row)
    return rows


def generate_missing(
    index_rows: list[dict[str, str]],
    priority: dict[tuple[str, str, str], dict[str, str]],
    *,
    max_workers: int,
    force: bool,
    object_parallel: bool = False,
) -> None:
    """Run the E199 upstream for each partner take that still misses aug motion.

    Takes are grouped by object; with object_parallel the groups run concurrently
    but each group runs its takes serially. Same-object takes must not run in
    parallel because pipeline.sh's sync_generated_object_model (cp -a) and
    ensure_g1_object_xml (first-write) race on the per-object model/XML files.
    """
    # group missing rows by partner base task (one retarget run yields all variants)
    takes: dict[str, dict[str, str]] = {}
    for row in index_rows:
        if row["partner_status"] != "missing":
            continue
        base = row["partner_base_task"]
        if base not in takes:
            src_case = row["source_case_id"]
            variant = row["source_variant"]
            man = priority[(row["object_key"], src_case, variant)]
            takes[base] = {
                "source_aug_task": man["target_task"],
                "partner_person": row["partner_person"],
                "object_key": row["object_key"],
            }
    if not takes:
        print("[generate] no missing partner takes -- nothing to do")
        return

    from collections import defaultdict

    groups: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for base, info in sorted(takes.items()):
        groups[info["object_key"]].append((base, info))
    print(f"[generate] {len(takes)} take(s) in {len(groups)} object group(s); "
          f"object_parallel={object_parallel}", flush=True)

    def run_group(obj: str, items: list[tuple[str, dict[str, str]]]) -> None:
        for base, info in items:
            meta = partner_meta(info["source_aug_task"], info["partner_person"])
            print(f"[{obj}] === partner retarget: {base} (person={meta['person']}) ===", flush=True)
            B.run_upstream(base, meta, force=force, max_workers=max_workers)
            trim_start, feasible = B.fixed_window_trim(base, meta, C.AUG_VARIANTS)
            print(f"[{obj}] {base}: trim_start={trim_start} feasible={feasible}", flush=True)

    if object_parallel and len(groups) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = [pool.submit(run_group, obj, items) for obj, items in groups.items()]
            for future in futures:
                future.result()
    else:
        for obj, items in groups.items():
            run_group(obj, items)


def finalize_index(
    index_rows: list[dict[str, str]],
    priority: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    """Re-resolve any previously-missing rows after generation."""
    out: list[dict[str, str]] = []
    for row in index_rows:
        if row["partner_status"] == "present":
            out.append(row)
            continue
        obj, src_case, variant = row["object_key"], row["source_case_id"], row["source_variant"]
        man = priority[(obj, src_case, variant)]
        meta = partner_meta(man["target_task"], row["partner_person"])
        found = resolve_partner(meta, row["partner_base_task"], variant)
        if found:
            row = {
                **row,
                "partner_trimmed_npz": C.rel(found[0]),
                "partner_omniretarget_output_npz": C.rel(found[1]),
                "partner_trim_window_json": C.rel(found[2]),
                "partner_status": "generated",
                "partner_generation_mode": "generate_e200_partner_aug",
            }
        else:
            row = {**row, "partner_status": "infeasible"}
        out.append(row)
    return out


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the partner motion index TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["object_key"], r["source_case_id"], r["source_variant"]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    """Resolve/generate augmented partner motions and write the index."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-tsv", type=Path, required=True,
                        help="the 5object_all selection.tsv (superset of aug cases)")
    parser.add_argument("--out-index", type=Path, required=True)
    parser.add_argument("--generate", action="store_true",
                        help="run the E199 upstream to build missing partner motions")
    parser.add_argument("--object-parallel", action="store_true",
                        help="run object groups concurrently (each group still serial)")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    priority = load_priority_index()
    aug_rows = selection_aug_rows(args.selection_tsv.expanduser())
    index = build_index(aug_rows, priority)
    n_present = sum(1 for r in index if r["partner_status"] == "present")
    print(f"[resolve] aug cases={len(index)}  present={n_present}  missing={len(index) - n_present}")

    if args.generate:
        generate_missing(index, priority, max_workers=args.max_workers, force=args.force,
                         object_parallel=args.object_parallel)
        index = finalize_index(index, priority)

    write_index(args.out_index.expanduser(), index)
    from collections import Counter
    counts = Counter(r["partner_status"] for r in index)
    print(f"[done] partner index -> {args.out_index}  status={dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
