#!/usr/bin/env python3
"""E208 P0: offline contract self-check (A1-A9).

Every assertion here can fail without touching a GPU, a conda env or the
upstream retargeter.  They exist because each one corresponds to a way E208
could silently produce data that is NOT comparable to the E206 baseline it is
measured against -- silent is the operative word; none of these would raise on
their own during a run.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E208/test_e208_contract.py
    ... --json-out PATH
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402


class Check:
    """One lettered assertion; collects detail even when it passes."""

    def __init__(self, code: str, title: str) -> None:
        self.code = code
        self.title = title
        self.detail: dict[str, Any] = {}
        self.failures: list[str] = []

    def require(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return ok

    @property
    def passed(self) -> bool:
        return not self.failures

    def payload(self) -> dict[str, Any]:
        return {
            "code": self.code, "title": self.title,
            "verdict": "pass" if self.passed else "fail",
            "failures": self.failures, "detail": self.detail,
        }


# --------------------------------------------------------------------------
def a1_registry(cases: list[dict[str, str]]) -> Check:
    chk = Check("A1", "registry derives exactly 22 RL_EXPORT_READY rows from the pinned E206 table")
    actual_sha = C.sha256(C.SOURCE_TSV)
    chk.detail["source_tsv"] = C.rel(C.SOURCE_TSV)
    chk.detail["source_sha256"] = actual_sha
    chk.require(
        actual_sha == C.EXPECTED_SOURCE_SHA256,
        f"source table sha256 {actual_sha} != pinned {C.EXPECTED_SOURCE_SHA256}",
    )

    chk.detail["n_cases"] = len(cases)
    chk.require(len(cases) == C.EXPECTED_CASES,
                f"{len(cases)} cases != expected {C.EXPECTED_CASES}")

    by_object: dict[str, int] = {}
    for case in cases:
        by_object[case["object_key"]] = by_object.get(case["object_key"], 0) + 1
    chk.detail["by_object"] = by_object
    chk.require(by_object == C.EXPECTED_CASES_BY_OBJECT,
                f"per-object counts {by_object} != pinned {C.EXPECTED_CASES_BY_OBJECT}")

    ids = [c["case_id"] for c in cases]
    chk.require(len(set(ids)) == len(ids), "duplicate case_id in registry")

    # every row must carry the orig artifacts C4 pairs against
    for case in cases:
        for key in ("orig_result_npz", "orig_scene_act", "orig_trajectory", "orig_contact_mask"):
            chk.require(bool(case[key]), f"{case['case_id']}: empty {key}")
        chk.require(case["orig_manual_use_decision"] == "USE",
                    f"{case['case_id']}: manual_use_decision={case['orig_manual_use_decision']!r} != USE")

    review_sha = C.sha256(C.MANUAL_REVIEW_TSV)
    chk.detail["manual_review_sha256"] = review_sha
    chk.require(review_sha == C.EXPECTED_MANUAL_REVIEW_SHA256,
                f"E206 manual review sha256 {review_sha} != pinned")
    return chk


def a2_base_tasks(cases: list[dict[str, str]]) -> Check:
    chk = Check("A2", "every base task dir exists with task_info.json + scene.xml; 21 v1 + 1 v2")
    variant_counts: dict[str, int] = {}
    for case in cases:
        base = case["base_target_task"]
        tdir = C.TASK_ROOT / base
        chk.require(tdir.is_dir(), f"{case['case_id']}: missing task dir {tdir}")
        chk.require((tdir / "task_info.json").is_file(), f"{base}: missing task_info.json")
        chk.require((tdir / "scene.xml").is_file(), f"{base}: missing scene.xml")
        # the dir name must agree with the retarget variant the manifest recorded
        recorded = case["source_retarget_variant_id"]
        chk.require(recorded in base,
                    f"{base}: dir name does not contain recorded variant {recorded!r}")
        variant_counts[recorded] = variant_counts.get(recorded, 0) + 1
    chk.detail["variant_counts"] = variant_counts
    chk.require(variant_counts == {"omnirt_v1": 21, "omnirt_v2": 1},
                f"variant split {variant_counts} != 21 v1 / 1 v2")

    source_v2 = sorted(c["case_id"] for c in cases if c["source_retarget_variant_id"] == "omnirt_v2")
    chk.detail["source_v2_cases"] = source_v2
    chk.require(tuple(source_v2) == C.SOURCE_V2_CASES,
                f"source-v2 cases {source_v2} != pinned {list(C.SOURCE_V2_CASES)}")
    return chk


def a3_aug_names(cases: list[dict[str, str]]) -> Check:
    chk = Check("A3", "aug_task_name is v1-aware, idempotent, and yields unique names")
    names: list[str] = []
    for case in cases:
        base = case["base_target_task"]
        eff = case["source_retarget_variant_id"]
        for variant, _holo in C.TRANS_VARIANTS:
            name = C.aug_task_name(base, variant, eff)
            names.append(name)
            # no implicit relabel: the name must carry the variant we asked for
            chk.require(eff in name, f"{name}: does not carry effective variant {eff}")
            other = "omnirt_v2" if eff == "omnirt_v1" else "omnirt_v1"
            chk.require(other not in name, f"{name}: leaks the other variant {other}")
            chk.require(name.endswith(f"__aug_{variant}"), f"{name}: bad suffix")
            # idempotent under the same effective variant
            renamed = C.aug_task_name(base, variant, eff)
            chk.require(renamed == name, f"{name}: aug_task_name is not idempotent")

    chk.detail["n_names"] = len(names)
    chk.require(len(names) == C.EXPECTED_CASES * len(C.TRANS_VARIANTS),
                f"{len(names)} aug names != {C.EXPECTED_CASES * len(C.TRANS_VARIANTS)}")
    chk.require(len(set(names)) == len(names), "duplicate aug task name")

    # the rescue path must be able to relabel a v1 base to v2 and back
    probe = "dcv3_omnirt_v1_ref_fk_desk007_20231030_028_p1"
    to_v2 = C.aug_task_name(probe, "trans0", "omnirt_v2")
    chk.detail["relabel_probe"] = to_v2
    chk.require(to_v2 == "dcv3_omnirt_v2_ref_fk_desk007_20231030_028_p1__aug_trans0",
                f"v1->v2 relabel produced {to_v2}")

    # E199's helper is the thing we must NOT be using -- prove it differs
    legacy = C.E199.aug_task_name(probe, "trans0")
    chk.detail["e199_legacy_name"] = legacy
    chk.require(legacy != C.aug_task_name(probe, "trans0", "omnirt_v1"),
                "e199_common.aug_task_name no longer differs from the v1-aware one; "
                "re-verify that the v1->v2 relabel hazard is gone before dropping A3")
    return chk


def a4_object_name_casing(cases: list[dict[str, str]]) -> Check:
    chk = Check("A4", "object_name is passed through verbatim (parallel_robot_retarget globs case-sensitively)")
    seen: dict[str, str] = {}
    for case in cases:
        meta = C.load_case_meta(case["base_target_task"])
        info = json.loads((C.TASK_ROOT / case["base_target_task"] / "task_info.json").read_text(encoding="utf-8"))
        chk.require(meta["object_name"] == info["object_name"],
                    f"{case['case_id']}: load_case_meta mangled object_name")
        seen[case["object_key"]] = meta["object_name"]
        # the holosoma task name embeds object_name; a lowercased copy would
        # make find_files() match zero npz files without erroring
        chk.require(meta["holosoma_task"].endswith(f"-{meta['object_name']}_with_obj"),
                    f"{case['case_id']}: holosoma_task does not embed object_name verbatim")
        for field in ("date", "seq", "person", "object_model_rel", "source_scene_task"):
            chk.require(bool(meta.get(field)), f"{case['case_id']}: empty {field}")
        # source_scene_task must be non-empty: pipeline.sh reads the case file
        # with IFS=$'\t' and an empty column shifts target_task (E200 note)
        chk.require("/" not in meta["source_scene_task"],
                    f"{case['case_id']}: source_scene_task looks like a path")
    chk.detail["object_name_by_key"] = seen
    # the known trap: desk021's object_name is capitalised
    chk.require(seen.get("desk021") == "Desk021",
                f"desk021 object_name is {seen.get('desk021')!r}; the casing trap moved -- re-check "
                "parallel_robot_retarget.find_files before relaxing this")
    return chk


def a5_module_shadowing() -> Check:
    chk = Check("A5", "load_e208_module cannot silently fall through to E199's same-named modules")
    siblings = ("build_augmented_tasks", "build_aug_manifest")

    # (a) the hazard is real: E199 ships the same module names and its dir is on sys.path
    e199_dir = C.EXPERIMENTS / "E199"
    collisions = [n for n in siblings if (e199_dir / f"{n}.py").is_file()]
    chk.detail["e199_collisions"] = collisions
    chk.detail["e199_on_syspath"] = str(e199_dir) in sys.path
    chk.require(bool(collisions) and str(e199_dir) in sys.path,
                "the shadowing hazard no longer exists; re-justify load_e208_module before removing it")

    # (b) the loader refuses names E208 does not own, instead of resolving elsewhere
    try:
        C.load_e208_module("definitely_not_an_e208_module")
        chk.require(False, "load_e208_module resolved a module E208 does not own")
    except FileNotFoundError:
        pass

    # (c) whatever E208 siblings already exist must resolve inside E208
    resolved: dict[str, str] = {}
    for name in siblings:
        if not (C.SCRIPT_DIR / f"{name}.py").is_file():
            continue
        module = C.load_e208_module(name)
        resolved[name] = module.__file__
        chk.require(Path(module.__file__).resolve().parent == C.SCRIPT_DIR,
                    f"{name} resolved to {module.__file__}, outside E208")
    chk.detail["resolved"] = resolved
    return chk


def a6_prg_only() -> Check:
    chk = Check("A6", "PRG-only arm; override payload taken from e206_common by value")
    chk.detail["arm_id"] = C.ARM_ID
    chk.detail["scene_name"] = C.SCENE_NAME
    chk.require(C.ARM_ID == "prg", f"ARM_ID={C.ARM_ID!r}")
    chk.require(C.SCENE_NAME == C.E206.SCENE_BY_ARM["prg"], "SCENE_NAME is not E206's PRG scene")

    # no noprg dimension anywhere the manifest or the output paths can see
    lowered = [f for f in C.FIELDS if "noprg" in f.lower()]
    chk.require(not lowered, f"manifest FIELDS carry a noprg dimension: {lowered}")
    sample = C.cem_out_dir("chair005_20231030_043_p1", "trans0")
    chk.detail["sample_out_dir"] = C.rel(sample)
    chk.require("noprg" not in str(sample).lower(), "cem_out_dir mentions noprg")
    chk.require(str(sample).endswith("_PRG"), "cem_out_dir is not PRG-tagged")

    # values, not a re-typed copy
    chk.require(C.PRG_OVERRIDES == C.E206.PRG_OVERRIDES, "PRG_OVERRIDES drifted from e206_common")
    chk.require(C.E163_HAND_GATE == C.E206.E163_HAND_GATE, "E163_HAND_GATE drifted from e206_common")
    chk.require(C.PRG_OVERRIDES.get("object_collision_sdf_mode") == "union",
                "object_collision_sdf_mode must be declared 'union' (spider/config.py:69-79 is fail-closed)")
    chk.require(C.PRG_OVERRIDES.get("scene_name") == C.SCENE_NAME,
                "PRG_OVERRIDES.scene_name disagrees with SCENE_NAME")
    chk.detail["n_prg_override_keys"] = len(C.PRG_OVERRIDES)
    chk.detail["n_hand_gate_keys"] = len(C.E163_HAND_GATE)

    # pair arithmetic comes from E206 too
    chk.require(C.expected_pair_counts(12)["prg"] == 216, "expected_pair_counts drifted")
    return chk


ENV_RE = re.compile(r"(?<![\w/.-])([A-Z][A-Z0-9_]*)=([^\s]*)")


def _stage2b_env(path: Path) -> dict[str, str]:
    """Pull the literal `env KEY=VALUE ...` assignments out of a run_stage2b script."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("env "):
            return dict(ENV_RE.findall(line))
    raise AssertionError(f"no `env ...` line in {path}")


def a7_retarget_env() -> Check:
    chk = Check("A7", "OMNIRT_V*_ENV spell out all six keys and match E206's recorded stage2b env")
    six = set(C.OMNIRT_V1_ENV)
    chk.detail["keys"] = sorted(six)
    chk.require(len(six) == 6, f"OMNIRT_V1_ENV has {len(six)} keys, expected 6")
    chk.require(set(C.OMNIRT_V2_ENV) == six, "V1/V2 env dicts do not cover the same keys")
    chk.require("REPLACE_WRIST_WITH_FINGERTIP" in six,
                "REPLACE_WRIST_WITH_FINGERTIP must be explicit: pipeline.sh:28 defaults it to 1, "
                "E206 used 0, and e199_common.OMNIRT_V2_ENV omits it")

    # e199's dict is the thing we must not have copied
    chk.detail["e199_v2_keys"] = sorted(C.E199.OMNIRT_V2_ENV)
    chk.require("REPLACE_WRIST_WITH_FINGERTIP" not in C.E199.OMNIRT_V2_ENV,
                "e199_common.OMNIRT_V2_ENV now sets REPLACE_WRIST_WITH_FINGERTIP; re-check A7's premise")

    observed: dict[str, dict[str, str]] = {}
    for variant, script in C.STAGE2B_SCRIPT.items():
        chk.require(script.is_file(), f"missing E206 stage2b script {script}")
        if not script.is_file():
            continue
        env = _stage2b_env(script)
        observed[variant] = {k: env.get(k, "<absent>") for k in sorted(six)}
        for key, expected in C.OMNIRT_ENV_BY_VARIANT[variant].items():
            got = env.get(key)
            chk.require(got == expected,
                        f"{variant}.{key}: E206 recorded {got!r}, E208 would send {expected!r}")
        for key, expected in C.PIPELINE_DATASET_ENV.items():
            got = env.get(key)
            chk.require(got == expected,
                        f"{variant}.{key}: E206 recorded {got!r}, E208 would send {expected!r}")
        # TARGET_VARIANT_ID is inert (pipeline.sh never reads it); assert it was
        # only ever a provenance label so E208 is free to omit it
        chk.require(env.get("TARGET_VARIANT_ID") == "ref_fk",
                    f"{variant}: unexpected TARGET_VARIANT_ID {env.get('TARGET_VARIANT_ID')!r}")
    chk.detail["observed"] = observed

    pipeline = C.PIPELINE_SH.read_text(encoding="utf-8")
    chk.require("TARGET_VARIANT_ID" not in pipeline,
                "pipeline.sh now reads TARGET_VARIANT_ID; ref_fk is no longer inert")
    chk.require('RETARGET_AUGMENTATION="${RETARGET_AUGMENTATION:-0}"' in pipeline,
                "pipeline.sh no longer exposes RETARGET_AUGMENTATION with a default of 0")
    chk.detail["pipeline_sh_sha256"] = C.sha256(C.PIPELINE_SH)
    return chk


def a8_budget() -> Check:
    chk = Check("A8", "CEM budget is read from E206's frozen admission block, never hardcoded")
    budget = C.frozen_budget()
    chk.detail["budget"] = budget
    chk.detail["source"] = C.rel(C.SOURCE_ADMISSION_JSON)
    chk.require(budget == C.EXPECTED_BUDGET,
                f"frozen budget {budget} != expected {C.EXPECTED_BUDGET}")
    payload = C.source_admission()
    for gate in ("A1_single_task", "A2_queue"):
        chk.detail[gate] = payload[gate]
    chk.require(C.PER_TASK_TIMEOUT_MIN > payload["A2_queue"]["per_task_bound_min"],
                f"per-task timeout {C.PER_TASK_TIMEOUT_MIN}min <= E206's per-task bound "
                f"{payload['A2_queue']['per_task_bound_min']}min; healthy long runs would be killed")

    # The queue E208 will actually face: 5 variants (not the 3 plan238 assumed)
    # over the 21 cases left after the chair005 exclusion.
    proj = C.queue_projection(C.eligible_potential())
    chk.detail["projected"] = proj
    chk.require(proj["verdict"] == "pass",
                f"projected worst-case {proj['bound_h']}h exceeds the "
                f"{proj['bar_hours']}h queue bar")
    return chk


def a9_namespace() -> Check:
    chk = Check("A9", "the E208 namespace is free, and the in-flight E207 experiment is untouched")
    ws = C.REPO / "workspace/core4d"

    plans = sorted(p.name for p in (ws / "plan").glob(f"{C.PLAN_SLOT}_*.md"))
    logs = sorted(p.name for p in (ws / "log").glob(f"{C.LOG_SLOT}_*.md"))
    chk.detail[f"plan_{C.PLAN_SLOT}"] = plans
    chk.detail[f"log_{C.LOG_SLOT}"] = logs
    chk.require(all("E208" in p for p in plans),
                f"plan slot {C.PLAN_SLOT} taken by something else: {plans}")
    chk.require(all("E208" in p for p in logs),
                f"log slot {C.LOG_SLOT} taken by something else: {logs}")

    tracker = (ws / "EXPERIMENT_TRACKER.md").read_text(encoding="utf-8")
    chk.require("R294" not in tracker, "run id R294 already appears in EXPERIMENT_TRACKER.md")

    strays = sorted(
        p.name for p in C.OVERRIDE_DIR.glob("core4d_E208_*.yaml")
        if not p.name.endswith("_lowgeom_PRG.yaml") and "__aug_" not in p.name
    )
    chk.detail["stray_e208_overrides"] = strays
    chk.require(not strays, f"unexpected core4d_E208_* overrides already present: {strays}")

    # E207 (bucket G1-only gravcomp) is running on this same branch and working
    # tree.  Its files are on the plan's zero-change list; prove we have not
    # touched them.
    e207_paths = [
        "workspace/core4d/scripts/experiments/E207",
        "workspace/core4d/plan/237_E207_bucket_g1only_gravcomp_plan.md",
    ]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *e207_paths],
        cwd=C.REPO, capture_output=True, text=True, check=False,
    ).stdout.strip()
    chk.detail["e207_git_status"] = dirty.splitlines()
    chk.require(not dirty, f"E207 files are modified in the working tree:\n{dirty}")

    stray_e207 = sorted(p.name for p in C.OVERRIDE_DIR.glob("core4d_E207_*.yaml"))
    chk.detail["n_e207_overrides"] = len(stray_e207)
    e207_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "examples/config/override/core4d_E207_*.yaml"],
        cwd=C.REPO, capture_output=True, text=True, check=False,
    ).stdout.strip()
    chk.require(not e207_dirty, f"E207 overrides modified:\n{e207_dirty}")
    return chk


# --------------------------------------------------------------------------
def a_extra_proxy_inheritance(cases: list[dict[str, str]]) -> Check:
    """Not lettered: the load-bearing assumption behind skipping proxy rebuild.

    E206 installed the hand-placed lowgeom boxes into the SOURCE TEMPLATE, so an
    aug task built from that template inherits them.  If that ever stops being
    true, P5 would build aug scenes against the original mesh hull and every
    contact metric would silently change.
    """
    chk = Check("A10", "source templates still carry E206's lowgeom box proxy (aug inherits it)")
    counts: dict[str, int] = {}
    for case in cases:
        meta = C.load_case_meta(case["base_target_task"])
        template = C.TASK_ROOT / meta["source_scene_task"] / "scene.xml"
        if not chk.require(template.is_file(), f"{case['case_id']}: missing template {template}"):
            continue
        root = ET.parse(template).getroot()
        geoms = [
            g for g in root.iter("geom")
            if (g.get("name") or "").startswith("object_collision")
        ]
        non_box = [g.get("name") for g in geoms if g.get("type") != "box"]
        chk.require(bool(geoms), f"{meta['source_scene_task']}: no object_collision geoms")
        chk.require(not non_box,
                    f"{meta['source_scene_task']}: non-box collision geoms {non_box} "
                    "(union SDF mode is fail-closed on these)")
        prev = counts.get(case["object_key"])
        chk.require(prev is None or prev == len(geoms),
                    f"{case['object_key']}: template box count differs between person1/person2 "
                    f"({prev} vs {len(geoms)})")
        counts[case["object_key"]] = len(geoms)
    chk.detail["object_geom_count"] = counts
    # E206's shipped proxy sizes (log295 C2): desk007 12, chair006 10, desk023 9,
    # desk021 5, chair005 2.  A drift here means the proxy was re-edited.
    expected = {"desk007": 12, "chair006": 10, "desk023": 9, "desk021": 5, "chair005": 2}
    chk.require(counts == expected, f"proxy box counts {counts} != E206's shipped {expected}")
    return chk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=C.CONTRACT_SELFCHECK_JSON)
    args = ap.parse_args()

    cases = C.load_e208_cases()
    checks = [
        a1_registry(cases),
        a2_base_tasks(cases),
        a3_aug_names(cases),
        a4_object_name_casing(cases),
        a5_module_shadowing(),
        a6_prg_only(),
        a7_retarget_env(),
        a8_budget(),
        a9_namespace(),
        a_extra_proxy_inheritance(cases),
    ]

    for chk in checks:
        mark = "PASS" if chk.passed else "FAIL"
        print(f"[{mark}] {chk.code}  {chk.title}")
        for failure in chk.failures:
            print(f"        - {failure}")

    failed = [c.code for c in checks if not c.passed]
    payload = {
        "experiment": C.EXP_ID,
        "run_id": C.RUN_ID,
        "plan_ref": C.PLAN_REF,
        "generated_at": C.now(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO,
                                   capture_output=True, text=True, check=False).stdout.strip(),
        "verdict": "pass" if not failed else "fail",
        "failed": failed,
        "checks": [c.payload() for c in checks],
    }
    C.write_json(args.json_out, payload)
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks pass -> {C.rel(args.json_out)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
