#!/usr/bin/env python3
"""Compare frozen E178/E187 user review annotations on E187 keep22."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
E187_EVAL = REPO / "workspace/core4d/results/E187/s6_downstream/eval/full"
E178_EVAL = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full"
E187_REVIEW = E187_EVAL / "user_manual_review_filled.tsv"
E178_REVIEW = E178_EVAL / "user_manual_review_filled.tsv"
E187_METRICS = E187_EVAL / "e187_case_metrics.tsv"
E178_METRICS = E178_EVAL / "e178_case_metrics.tsv"
OUT_TSV = E187_EVAL / "e187_vs_e178_manual_review_comparison.tsv"
OUT_JSON = E187_EVAL / "manual_review_comparison_summary.json"
OUT_MD = E187_EVAL / "manual_review_comparison_summary.md"

DECISIVE = {"USE", "DO_NOT_USE"}
DECISIONS = DECISIVE | {"PENDING"}
QUALITY_ORDER = {
    "UNUSABLE": 0,
    "MAJOR_DEFECT": 1,
    "MINOR_ACCEPTABLE": 2,
    "CLEAN": 3,
}
OUTPUT_FIELDS = [
    "ordinal",
    "case_id",
    "object_key",
    "comparison_scope",
    "manual_transition",
    "manual_outcome",
    "e178_annotation_present",
    "e178_decision",
    "e187_decision",
    "e178_quality",
    "e187_quality",
    "quality_transition",
    "quality_direction",
    "e178_note",
    "e187_note",
    "e178_numeric_pass",
    "e187_numeric_pass",
    "numeric_transition",
    "e178_numeric_failure_modes",
    "e187_numeric_failure_modes",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 TSV file."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def keyed(rows: list[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    """Index rows by unique case ID."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id:
            raise ValueError(f"{source}: empty case_id")
        if case_id in result:
            raise ValueError(f"{source}: duplicate case_id={case_id}")
        result[case_id] = row
    return result


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(annotation: dict[str, str] | None) -> str:
    """Normalize missing or empty decisions to PENDING."""
    if annotation is None:
        return "PENDING"
    value = annotation.get("manual_use_decision", "") or "PENDING"
    if value not in DECISIONS:
        raise ValueError(f"unsupported manual decision={value!r}")
    return value


def validate_quality(case_id: str, annotation: dict[str, str]) -> None:
    """Validate the decision-to-quality-label contract."""
    dec = decision(annotation)
    quality = annotation.get("manual_quality_label", "")
    if dec == "USE" and quality not in {"CLEAN", "MINOR_ACCEPTABLE"}:
        raise ValueError(f"{case_id}: USE has invalid quality={quality!r}")
    if dec == "DO_NOT_USE" and quality not in {"MAJOR_DEFECT", "UNUSABLE"}:
        raise ValueError(f"{case_id}: DO_NOT_USE has invalid quality={quality!r}")
    if dec == "PENDING" and quality:
        raise ValueError(f"{case_id}: PENDING must not have quality={quality!r}")


def bool_text(value: str) -> str:
    """Normalize a serialized boolean without coercing unknown values."""
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid bool text={value!r}")
    return normalized


def manual_outcome(old: str, new: str) -> tuple[str, str]:
    """Classify one paired manual transition."""
    if old not in DECISIVE or new not in DECISIVE:
        if old == "PENDING" and new == "PENDING":
            scope = "BOTH_PENDING"
        elif old == "PENDING":
            scope = "E178_PENDING"
        else:
            scope = "E187_PENDING"
        return scope, "PENDING_COMPARISON"
    outcomes = {
        ("USE", "USE"): "RETAINED_USE",
        ("USE", "DO_NOT_USE"): "REGRESSED",
        ("DO_NOT_USE", "USE"): "RECOVERED",
        ("DO_NOT_USE", "DO_NOT_USE"): "RETAINED_DNU",
    }
    return "COMMON_DECISIVE", outcomes[(old, new)]


def exact_mcnemar_p(recovered: int, regressed: int) -> float:
    """Compute the two-sided exact McNemar binomial p-value."""
    discordant = recovered + regressed
    if discordant == 0:
        return 1.0
    tail = min(recovered, regressed)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def numeric_alignment(
    authority: list[str],
    annotations: dict[str, dict[str, str]],
    metrics: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Measure frozen numeric-gate alignment with decisive manual labels."""
    decisive_cases = [
        case_id
        for case_id in authority
        if decision(annotations.get(case_id)) in DECISIVE
    ]
    use_cases = [
        case_id
        for case_id in decisive_cases
        if decision(annotations.get(case_id)) == "USE"
    ]
    dnu_cases = [
        case_id
        for case_id in decisive_cases
        if decision(annotations.get(case_id)) == "DO_NOT_USE"
    ]
    numeric_pass = {
        case_id: bool_text(metrics[case_id]["numeric_release_pass"]) == "true"
        for case_id in decisive_cases
    }
    false_reject = [case_id for case_id in use_cases if not numeric_pass[case_id]]
    false_accept = [case_id for case_id in dnu_cases if numeric_pass[case_id]]
    return {
        "decisive_count": len(decisive_cases),
        "manual_use_count": len(use_cases),
        "manual_dnu_count": len(dnu_cases),
        "numeric_pass_on_decisive_count": sum(numeric_pass.values()),
        "manual_use_recall": (
            sum(numeric_pass[case_id] for case_id in use_cases) / len(use_cases)
            if use_cases
            else None
        ),
        "manual_dnu_rejection": (
            sum(not numeric_pass[case_id] for case_id in dnu_cases) / len(dnu_cases)
            if dnu_cases
            else None
        ),
        "numeric_false_reject_manual_use": false_reject,
        "numeric_false_accept_manual_dnu": false_accept,
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the deterministic paired comparison TSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, object]) -> str:
    """Render a concise human-readable summary."""
    paired = summary["common_decisive"]
    e178 = summary["decision_counts"]["E178_keep22"]
    e187 = summary["decision_counts"]["E187_keep22"]
    lines = [
        "# E187 vs E178 人工终审配对比较",
        "",
        "## 结论",
        "",
        f"- E187：USE={e187['USE']}、DO_NOT_USE={e187['DO_NOT_USE']}、PENDING={e187['PENDING']}；最终裁决 {summary['coverage']['E187_decisive']}/22。",
        f"- E178 keep22：USE={e178['USE']}、DO_NOT_USE={e178['DO_NOT_USE']}、PENDING={e178['PENDING']}；最终裁决 {summary['coverage']['E178_decisive']}/22。",
        f"- 双方均最终裁决的 {paired['count']} 条：USE→USE={paired['transitions']['USE->USE']}、DNU→USE={paired['transitions']['DO_NOT_USE->USE']}、USE→DNU={paired['transitions']['USE->DO_NOT_USE']}、DNU→DNU={paired['transitions']['DO_NOT_USE->DO_NOT_USE']}。",
        f"- 可比子集人工 USE 从 {paired['e178_use_count']}/{paired['count']} 变为 {paired['e187_use_count']}/{paired['count']}，净变化 {paired['use_count_delta']:+d}（{paired['use_rate_delta_pp']:+.1f}pp）；McNemar exact p={paired['mcnemar_exact_two_sided_p']:.3f}。",
        "- 方向上 E187 人工可用度改善且没有 USE→DNU，但样本和未决项使其不足以宣称统计显著或22条终审完成。",
        "",
        "## 人工迁移",
        "",
        "| 类别 | Case |",
        "|---|---|",
        f"| DNU→USE | {', '.join(paired['recovered_cases']) or '—'} |",
        f"| USE→DNU | {', '.join(paired['regressed_cases']) or '—'} |",
        f"| USE→USE | {', '.join(paired['retained_use_cases']) or '—'} |",
        f"| DNU→DNU | {', '.join(paired['retained_dnu_cases']) or '—'} |",
        "",
        "## 未决项",
        "",
        f"- E178未决：{', '.join(summary['pending']['E178']) or '—'}",
        f"- E187未决：{', '.join(summary['pending']['E187']) or '—'}",
        "",
        "## Numeric 与人工",
        "",
        "| 实验 | 人工USE recall | 人工DNU rejection | numeric误拒USE | numeric误收DNU |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp in ("E178", "E187"):
        item = summary["numeric_alignment"][exp]
        lines.append(
            f"| {exp} | {item['manual_use_recall']:.3f} | {item['manual_dnu_rejection']:.3f} | {len(item['numeric_false_reject_manual_use'])} | {len(item['numeric_false_accept_manual_dnu'])} |"
        )
    lines.extend(
        [
            "",
            "E187 numeric 12门比人工更保守，尤其lower-body gate拒绝了多条人工USE；人工结果不支持把numeric `6/22 vs 8/22`直接解释成可用度下降。",
            "",
            "## 治理",
            "",
            "E187 C9保持technical `FAIL`、progression authority=`USER_WAIVED`。三条E187 PENDING未补猜，当前仍是`PENDING_USER_REVIEW`，不是人工终审完成。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Build and validate all manual-review comparison artifacts."""
    inputs = (E187_REVIEW, E178_REVIEW, E187_METRICS, E178_METRICS)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    e187_review_rows = read_tsv(E187_REVIEW)
    e178_review_rows = read_tsv(E178_REVIEW)
    e187_metrics_rows = read_tsv(E187_METRICS)
    e178_metrics_rows = read_tsv(E178_METRICS)
    e187_reviews = keyed(e187_review_rows, "E187 review")
    e178_reviews = keyed(e178_review_rows, "E178 review")
    e187_metrics = keyed(e187_metrics_rows, "E187 metrics")
    e178_metrics = keyed(e178_metrics_rows, "E178 metrics")

    authority = [row["case_id"] for row in e187_metrics_rows]
    if len(authority) != 22 or len(set(authority)) != 22:
        raise ValueError("E187 metrics must define exactly 22 unique authority cases")
    authority_set = set(authority)
    if set(e187_reviews) != authority_set:
        raise ValueError(
            "E187 review case set must exactly match E187 metrics authority"
        )
    if not authority_set <= set(e178_metrics):
        raise ValueError("E178 metrics missing E187 authority cases")
    for case_id, annotation in e187_reviews.items():
        validate_quality(case_id, annotation)
    for case_id, annotation in e178_reviews.items():
        validate_quality(case_id, annotation)

    comparison_rows: list[dict[str, object]] = []
    transitions: Counter[str] = Counter()
    object_transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    outcome_cases: defaultdict[str, list[str]] = defaultdict(list)
    quality_directions: Counter[str] = Counter()
    quality_cases: defaultdict[str, list[str]] = defaultdict(list)
    for ordinal, case_id in enumerate(authority, start=1):
        old_ann = e178_reviews.get(case_id)
        new_ann = e187_reviews[case_id]
        old = decision(old_ann)
        new = decision(new_ann)
        scope, outcome = manual_outcome(old, new)
        transition = f"{old}->{new}"
        old_quality = old_ann.get("manual_quality_label", "") if old_ann else ""
        new_quality = new_ann.get("manual_quality_label", "")
        quality_direction = "NOT_COMPARABLE"
        if scope == "COMMON_DECISIVE":
            transitions[transition] += 1
            object_transitions[e187_metrics[case_id]["object_key"]][transition] += 1
            outcome_cases[outcome].append(case_id)
            if old_quality not in QUALITY_ORDER or new_quality not in QUALITY_ORDER:
                raise ValueError(f"{case_id}: missing comparable quality labels")
            delta = QUALITY_ORDER[new_quality] - QUALITY_ORDER[old_quality]
            quality_direction = (
                "IMPROVED" if delta > 0 else "REGRESSED" if delta < 0 else "SAME"
            )
            quality_directions[quality_direction] += 1
            quality_cases[quality_direction].append(case_id)
        comparison_rows.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "object_key": e187_metrics[case_id]["object_key"],
                "comparison_scope": scope,
                "manual_transition": transition,
                "manual_outcome": outcome,
                "e178_annotation_present": str(old_ann is not None).lower(),
                "e178_decision": old,
                "e187_decision": new,
                "e178_quality": old_quality,
                "e187_quality": new_quality,
                "quality_transition": f"{old_quality or 'PENDING'}->{new_quality or 'PENDING'}",
                "quality_direction": quality_direction,
                "e178_note": old_ann.get("manual_review_note", "") if old_ann else "",
                "e187_note": new_ann.get("manual_review_note", ""),
                "e178_numeric_pass": bool_text(
                    e178_metrics[case_id]["numeric_release_pass"]
                ),
                "e187_numeric_pass": bool_text(
                    e187_metrics[case_id]["numeric_release_pass"]
                ),
                "numeric_transition": (
                    f"{bool_text(e178_metrics[case_id]['numeric_release_pass'])}"
                    f"->{bool_text(e187_metrics[case_id]['numeric_release_pass'])}"
                ),
                "e178_numeric_failure_modes": e178_metrics[case_id].get(
                    "numeric_failure_modes", ""
                ),
                "e187_numeric_failure_modes": e187_metrics[case_id].get(
                    "numeric_failure_modes", ""
                ),
            }
        )

    common_count = sum(transitions.values())
    recovered = transitions["DO_NOT_USE->USE"]
    regressed = transitions["USE->DO_NOT_USE"]
    e178_use = transitions["USE->USE"] + regressed
    e187_use = transitions["USE->USE"] + recovered
    decision_counts = {
        "E178_keep22": dict(
            Counter(decision(e178_reviews.get(case_id)) for case_id in authority)
        ),
        "E187_keep22": dict(
            Counter(decision(e187_reviews.get(case_id)) for case_id in authority)
        ),
    }
    for counts in decision_counts.values():
        for value in ("USE", "DO_NOT_USE", "PENDING"):
            counts.setdefault(value, 0)

    summary: dict[str, object] = {
        "schema": "e187_vs_e178_manual_review_comparison_v1",
        "status": "PENDING_USER_REVIEW"
        if decision_counts["E187_keep22"]["PENDING"]
        else "USER_REVIEW_COMPLETE",
        "authority": {
            "case_count": len(authority),
            "E187_review_path": str(E187_REVIEW.relative_to(REPO)),
            "E187_review_sha256": sha256(E187_REVIEW),
            "E178_review_path": str(E178_REVIEW.relative_to(REPO)),
            "E178_review_sha256": sha256(E178_REVIEW),
            "E187_metrics_sha256": sha256(E187_METRICS),
            "E178_metrics_sha256": sha256(E178_METRICS),
        },
        "coverage": {
            "E187_annotation_rows": len(e187_review_rows),
            "E187_decisive": sum(
                decision(e187_reviews.get(case_id)) in DECISIVE for case_id in authority
            ),
            "E178_annotation_rows_in_keep22": sum(
                case_id in e178_reviews for case_id in authority
            ),
            "E178_decisive": sum(
                decision(e178_reviews.get(case_id)) in DECISIVE for case_id in authority
            ),
            "common_decisive": common_count,
        },
        "decision_counts": decision_counts,
        "pending": {
            "E178": [
                case_id
                for case_id in authority
                if decision(e178_reviews.get(case_id)) == "PENDING"
            ],
            "E187": [
                case_id
                for case_id in authority
                if decision(e187_reviews.get(case_id)) == "PENDING"
            ],
        },
        "common_decisive": {
            "count": common_count,
            "transitions": {
                "USE->USE": transitions["USE->USE"],
                "USE->DO_NOT_USE": regressed,
                "DO_NOT_USE->USE": recovered,
                "DO_NOT_USE->DO_NOT_USE": transitions["DO_NOT_USE->DO_NOT_USE"],
            },
            "e178_use_count": e178_use,
            "e187_use_count": e187_use,
            "use_count_delta": e187_use - e178_use,
            "use_rate_delta_pp": 100.0 * (e187_use - e178_use) / common_count,
            "mcnemar_discordant_count": recovered + regressed,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(recovered, regressed),
            "recovered_cases": outcome_cases["RECOVERED"],
            "regressed_cases": outcome_cases["REGRESSED"],
            "retained_use_cases": outcome_cases["RETAINED_USE"],
            "retained_dnu_cases": outcome_cases["RETAINED_DNU"],
        },
        "object_transitions_common_decisive": {
            object_key: dict(counts)
            for object_key, counts in sorted(object_transitions.items())
        },
        "quality_common_decisive": {
            "direction_counts": dict(quality_directions),
            "improved_cases": quality_cases["IMPROVED"],
            "regressed_cases": quality_cases["REGRESSED"],
            "same_cases": quality_cases["SAME"],
        },
        "numeric_alignment": {
            "E178": numeric_alignment(authority, e178_reviews, e178_metrics),
            "E187": numeric_alignment(authority, e187_reviews, e187_metrics),
        },
        "governance": {
            "E187_c9_technical": "FAIL",
            "E187_progression_authority": "USER_WAIVED",
            "full_case_count": 22,
            "manual_review_complete": decision_counts["E187_keep22"]["PENDING"] == 0,
        },
    }

    write_tsv(OUT_TSV, comparison_rows)
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(summary), encoding="utf-8")

    if len(read_tsv(OUT_TSV)) != 22:
        raise RuntimeError("comparison TSV row count mismatch")
    print(f"E187_MANUAL_COMPARISON=PASS rows=22 common={common_count}")
    print(
        f"RECOVERED={recovered} REGRESSED={regressed} PENDING_E187={decision_counts['E187_keep22']['PENDING']}"
    )
    print(f"TSV={OUT_TSV}")
    print(f"SUMMARY={OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
