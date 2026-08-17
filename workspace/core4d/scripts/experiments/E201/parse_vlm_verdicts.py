#!/usr/bin/env python3
"""E201 · parse VLM pre-screen predictions -> verdict TSV, and validate vs human.

Reads call_api output (generate_predictions.jsonl: predict[0] = model text,
metadata echoed), extracts the strict-JSON verdict, and writes a verdict TSV
(one row per rollout: vlm_use / vlm_quality / vlm_failure / vlm_worst / vlm_note,
or vlm_parse_error). Then, for rollouts that also have a human label in
user_manual_review_filled.tsv, prints VLM↔human agreement + false-accept count
(the key metric: VLM says USE but human says DO_NOT_USE).

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E201/parse_vlm_verdicts.py --exp E199
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
VLM_DIR = REPO / "workspace/core4d/results/E201/vlm_review"
GT_TSV = REPO / "workspace/core4d/results/E199/s6_downstream/eval/fullscale_augmentation/user_manual_review_filled.tsv"

VALID_USE = {"USE", "DO_NOT_USE"}
VALID_QUAL = {"CLEAN", "MINOR_ACCEPTABLE", "MAJOR_DEFECT", "UNUSABLE"}


def family_key(case_id: str) -> str:
    return re.sub(r"_p(\d)$", r"_person\1", case_id)


def extract_text(predict) -> str:
    """predict is [response]; response may be a str or an openai-style dict."""
    if not predict:
        return ""
    r = predict[0]
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        for k in ("content", "text", "message"):
            v = r.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("content"), str):
                return v["content"]
    return str(r)


def parse_json_verdict(text: str) -> dict | None:
    """Find and parse the first JSON object in the model output."""
    if not text:
        return None
    # strip ```json fences if present
    text = re.sub(r"```(?:json)?", "", text)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199")
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pred_path = args.predictions or (VLM_DIR / "out" / args.exp / "0" / "generate_predictions.jsonl")
    out = args.out or (VLM_DIR / "verdicts" / f"{args.exp}_vlm_verdicts.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)

    verdicts = []
    n_parse_err = 0
    with pred_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            text = extract_text(rec.get("predict", []))
            v = parse_json_verdict(text)
            row = {
                "case_id": md.get("case_id", ""), "aug_variant": md.get("aug_variant", ""),
                "object_key": md.get("object_key", ""), "layer": md.get("layer", ""),
                "family_flag": md.get("family_flag", ""), "n_frames": md.get("n_frames", ""),
            }
            if v is None:
                n_parse_err += 1
                row.update({"vlm_use": "", "vlm_quality": "", "vlm_failure": "",
                            "vlm_worst": "", "vlm_note": "", "vlm_parse_error": "1",
                            "raw": text[:300].replace("\t", " ").replace("\n", " ")})
            else:
                use = str(v.get("use_decision", "")).upper().strip()
                qual = str(v.get("quality_label", "")).upper().strip()
                row.update({
                    "vlm_use": use if use in VALID_USE else f"?{use}",
                    "vlm_quality": qual if qual in VALID_QUAL else f"?{qual}",
                    "vlm_failure": ",".join(map(str, v.get("failure_taxonomy", []) or [])),
                    "vlm_worst": ",".join(map(str, v.get("worst_frames", []) or [])),
                    "vlm_note": str(v.get("overall_note", "")).replace("\t", " ").replace("\n", " "),
                    "vlm_parse_error": "", "raw": "",
                })
            verdicts.append(row)

    cols = ["case_id", "aug_variant", "object_key", "layer", "family_flag", "n_frames",
            "vlm_use", "vlm_quality", "vlm_failure", "vlm_worst", "vlm_note",
            "vlm_parse_error", "raw"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(verdicts)
    print(f"[done] wrote {out} ({len(verdicts)} verdicts; parse_error={n_parse_err})")

    # ---- validate vs human ground truth ----
    if not GT_TSV.is_file():
        print("[validate] no ground-truth file, skip")
        return 0
    gt = {}
    for r in csv.DictReader(GT_TSV.open(), delimiter="\t"):
        cid = r["case_id"].strip(); dec = r["manual_use_decision"].strip()
        if "#" in cid and dec in VALID_USE:
            b, v = cid.split("#", 1)
            gt[(family_key(b), v)] = dec
    conf = defaultdict(Counter)
    matched = 0
    for row in verdicts:
        k = (family_key(row["case_id"]), row["aug_variant"])
        if k not in gt or row["vlm_use"] not in VALID_USE:
            continue
        matched += 1
        conf[row["vlm_use"]][gt[k]] += 1
    if not matched:
        print("[validate] 0 rollouts overlap human labels")
        return 0
    fa = conf["USE"]["DO_NOT_USE"]        # VLM says USE, human DO_NOT_USE (critical)
    fr = conf["DO_NOT_USE"]["USE"]        # VLM says DO_NOT_USE, human USE
    tu = conf["USE"]["USE"]; tr = conf["DO_NOT_USE"]["DO_NOT_USE"]
    print(f"\n=== VLM vs human (n={matched} overlap in L2/L3-review) ===")
    print(f"  agreement          : {(tu+tr)}/{matched} = {(tu+tr)/matched:.1%}")
    print(f"  VLM USE & human USE        : {tu}")
    print(f"  VLM DO_NOT_USE & human DNU : {tr}")
    print(f"  FALSE ACCEPT (VLM USE, human DO_NOT_USE): {fa}  <-- key metric")
    print(f"  false reject (VLM DO_NOT_USE, human USE): {fr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
