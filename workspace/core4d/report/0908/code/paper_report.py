#!/usr/bin/env python3
"""Report writers for the SPIDER-CEM vs OmniRetarget paper table.

Consumes the per-(case, method) records produced by ``gen_paper_results.py`` and emits
Markdown / xlsx / LaTeX / TSV plus a provenance JSON.  Granularity: per-case rows,
per-object summary (mean ± std + worst case), and an overall summary.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from gen_paper_results import (
    METRICS, METRIC_BY_KEY, OMNI_KEYS, SPIDER_KEYS, OBJECT_ORDER, OUT,
)

CONTACT_KEYS = [m["key"] for m in METRICS if m["kind"] == "pct"]
TRACK_KEYS = ["track_root_pos", "track_root_ori", "track_eef_pos", "track_eef_ori",
              "track_obj_pos", "track_obj_ori"]
HEALTH_KEYS = ["fall_flag", "body_z", "ankle_jerk", "obj_speed", "foot_slip"]

BY_CASE_TSV = OUT / "paper_results_by_case.tsv"
MD = OUT / "paper_results.md"
XLSX = OUT / "paper_results.xlsx"
TEX = OUT / "paper_results.tex"
PROV = OUT / "provenance.json"

RESOLVED = "ok"


# --------------------------------------------------------------------------------------
# value formatting
# --------------------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_val(key: str, v: Any) -> str:
    if not _finite(v):
        return "—"
    v = float(v)
    kind = METRIC_BY_KEY[key]["kind"]
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "cm":
        return f"{v:.1f}"
    if kind == "deg":
        return f"{v:.1f}"
    if kind == "m":
        return f"{v:.3f}"
    if kind == "mps":
        return f"{v:.3f}"
    if kind == "jerk":
        return f"{v:.0f}"
    if kind == "flag":
        return f"{100 * v:.0f}%"
    return f"{v:.3f}"


def unit(key: str) -> str:
    return {"pct": "", "cm": " cm", "deg": "°", "m": " m", "mps": " m/s",
            "jerk": " m/s³", "flag": ""}[METRIC_BY_KEY[key]["kind"]]


def header_label(key: str) -> str:
    m = METRIC_BY_KEY[key]
    arrow = "↑" if m["dir"] == "higher" else "↓"
    return f"{m['label']}{unit(key)} {arrow}"


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

def method_metric(records: list[dict], method: str) -> dict[str, dict[str, float]]:
    """Return {case_id: {metric_key: value}} for resolved rows of a method."""
    out: dict[str, dict[str, float]] = {}
    for r in records:
        if r["method"] != method or r["status"] != RESOLVED:
            continue
        out[r["case_id"]] = {k: float(v) for k, v in r.get("metrics", {}).items() if _finite(v)}
    return out


def worst(key: str, values: list[float]) -> float:
    if not values:
        return math.nan
    return min(values) if METRIC_BY_KEY[key]["dir"] == "higher" else max(values)


def agg(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if _finite(v)]
    if not vals:
        return math.nan, math.nan, 0
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def summarize(cases_metrics: dict[str, dict[str, float]], keys: list[str],
              case_ids: list[str]) -> dict[str, dict[str, float]]:
    """{metric_key: {mean, std, worst, n}} over the given case_ids."""
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [cases_metrics[c][key] for c in case_ids
                if c in cases_metrics and key in cases_metrics[c] and _finite(cases_metrics[c][key])]
        mean, std, n = agg(vals)
        out[key] = {"mean": mean, "std": std, "worst": worst(key, vals), "n": n}
    return out


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------

def ms(cell: dict[str, float], key: str) -> str:
    if cell["n"] == 0:
        return "—"
    return f"{fmt_val(key, cell['mean'])} ± {fmt_val(key, cell['std'])}"


def write_markdown(records, cases, spider_m, omni_m) -> None:
    objects = [o for o in OBJECT_ORDER if any(object_of(c) == o for c in cases)]
    L: list[str] = []
    L += ["# SPIDER-CEM vs OmniRetarget — paper metrics", "",
          f"_{len(cases)} cases across {len(objects)} object groups; "
          "metrics from the shared `core_metrics.evaluate_sequence`._", ""]

    # ---- overall ----
    all_cases = list(cases)
    sp_all = summarize(spider_m, SPIDER_KEYS, all_cases)
    om_all = summarize(omni_m, OMNI_KEYS, all_cases)
    L += ["## Overall (mean ± std over all resolved cases)", "",
          "### Contact / penetration — SPIDER-CEM vs OmniRetarget", "",
          "| metric | SPIDER-CEM | OmniRetarget |", "| --- | ---: | ---: |"]
    for k in CONTACT_KEYS:
        sp = f"{ms(sp_all[k], k)} (n={sp_all[k]['n']})"
        om = f"{ms(om_all[k], k)} (n={om_all[k]['n']})" if k in om_all else "—"
        L.append(f"| {header_label(k)} | {sp} | {om} |")
    L += ["", "### SPIDER-CEM tracking / health / foot-slide", "",
          "| metric | mean ± std | worst | n |", "| --- | ---: | ---: | ---: |"]
    for k in TRACK_KEYS + HEALTH_KEYS:
        c = sp_all[k]
        L.append(f"| {header_label(k)} | {ms(c, k)} | {fmt_val(k, c['worst'])} | {c['n']} |")

    # ---- per object ----
    L += ["", "## Per-object summary", ""]
    L += ["### Contact / penetration (SPIDER-CEM ‖ OmniRetarget, mean±std)", "",
          "| object | n | " + " | ".join(header_label(k) for k in CONTACT_KEYS) + " |",
          "| --- | ---: | " + " | ".join("---:" for _ in CONTACT_KEYS) + " |"]
    for o in objects:
        ocases = [c for c in cases if object_of(c) == o]
        sp = summarize(spider_m, CONTACT_KEYS, ocases)
        om = summarize(omni_m, CONTACT_KEYS, ocases)
        cells = [f"{ms(sp[k], k)} ‖ {ms(om[k], k)}" for k in CONTACT_KEYS]
        L.append(f"| {o} | {len(ocases)} | " + " | ".join(cells) + " |")

    L += ["", "### SPIDER-CEM tracking / health / foot-slide (mean±std)", "",
          "| object | n | " + " | ".join(header_label(k) for k in TRACK_KEYS + HEALTH_KEYS) + " |",
          "| --- | ---: | " + " | ".join("---:" for _ in TRACK_KEYS + HEALTH_KEYS) + " |"]
    for o in objects:
        ocases = [c for c in cases if object_of(c) == o]
        sp = summarize(spider_m, TRACK_KEYS + HEALTH_KEYS, ocases)
        cells = [ms(sp[k], k) for k in TRACK_KEYS + HEALTH_KEYS]
        L.append(f"| {o} | {len(ocases)} | " + " | ".join(cells) + " |")

    # ---- per case ----
    L += ["", "## Per-case detail", "",
          "SPIDER-CEM rows carry all metrics; OmniRetarget rows carry contact/penetration only.", "",
          "| case | method | " + " | ".join(header_label(k) for k in SPIDER_KEYS) + " |",
          "| --- | --- | " + " | ".join("---:" for _ in SPIDER_KEYS) + " |"]
    for c in cases:
        for method, mp, keys in (("SPIDER-CEM", spider_m, SPIDER_KEYS), ("OmniRetarget", omni_m, OMNI_KEYS)):
            vals = mp.get(c, {})
            cells = []
            for k in SPIDER_KEYS:
                cells.append(fmt_val(k, vals[k]) if (method == "SPIDER-CEM" or k in keys) and k in vals else "")
            status = "" if c in mp else " *(unresolved)*"
            L.append(f"| {c}{status} | {method} | " + " | ".join(cells) + " |")

    # ---- coverage ----
    unresolved_sp = [r["case_id"] for r in records if r["method"] == "SPIDER-CEM" and r["status"] != RESOLVED]
    unresolved_om = [r["case_id"] for r in records if r["method"] == "OmniRetarget" and r["status"] != RESOLVED]
    L += ["", "## Coverage & caveats", "",
          f"- SPIDER-CEM resolved: {len(cases) - len(unresolved_sp)}/{len(cases)}. "
          f"Unresolved: {unresolved_sp or 'none'}",
          f"- OmniRetarget resolved: {len(cases) - len(unresolved_om)}/{len(cases)}. "
          f"Unresolved: {unresolved_om or 'none'}",
          "- Contact/penetration keys: raw contact "
          "`hand_object_physics_contact_in_mask_frac`, contact@3mm "
          "`hand_object_physics_contact_3mm_in_mask_frac`, penetration@3mm "
          "`hand_object_physics_penetration_3mm_frame_frac`, geom@2mm "
          "`hand_geom_penetration_2mm_frac`.",
          "- OmniRetarget physics metrics come from MuJoCo position-servo replay of its "
          "kinematic trajectory under the same `scene_act` (E197/E109 protocol).",
          "- ↑ higher is better, ↓ lower is better. `fall`/flags shown as % of cases.", ""]
    MD.write_text("\n".join(L), encoding="utf-8")


def object_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


# --------------------------------------------------------------------------------------
# tsv
# --------------------------------------------------------------------------------------

def write_by_case_tsv(records) -> None:
    cols = ["case_id", "object_key", "method", "status"] + SPIDER_KEYS + ["notes"]
    lines = ["\t".join(cols)]
    for r in records:
        m = r.get("metrics", {})
        row = [r["case_id"], r["object_key"], r["method"], r["status"]]
        for k in SPIDER_KEYS:
            v = m.get(k)
            row.append(f"{float(v):.6g}" if _finite(v) else "")
        row.append(";".join(r.get("notes", [])))
        lines.append("\t".join(row))
    BY_CASE_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# latex
# --------------------------------------------------------------------------------------

def _tex_num(key: str, cell: dict[str, float]) -> str:
    if cell["n"] == 0:
        return "--"
    return f"{fmt_val(key, cell['mean'])}".replace("%", r"\%").replace("±", r"$\pm$")


def write_latex(cases, spider_m, omni_m) -> None:
    objects = [o for o in OBJECT_ORDER if any(object_of(c) == o for c in cases)]
    L = [r"% Auto-generated: SPIDER-CEM vs OmniRetarget paper tables", ""]

    # Table 1: contact/penetration, SPIDER vs Omni, per object + overall (means).
    L += [r"\begin{table}[t]", r"\centering",
          r"\caption{Contact and penetration: SPIDER-CEM vs OmniRetarget (per-object mean).}",
          r"\begin{tabular}{l r " + " ".join("r" for _ in CONTACT_KEYS) + r"}", r"\toprule"]
    head = ["Object", "n"] + [f"{METRIC_BY_KEY[k]['label']} (S/O)".replace("%", r"\%") for k in CONTACT_KEYS]
    L.append(" & ".join(head) + r" \\")
    L.append(r"\midrule")
    for o in objects + ["ALL"]:
        ocases = cases if o == "ALL" else [c for c in cases if object_of(c) == o]
        sp = summarize(spider_m, CONTACT_KEYS, ocases)
        om = summarize(omni_m, CONTACT_KEYS, ocases)
        cells = [o, str(len(ocases))]
        for k in CONTACT_KEYS:
            cells.append(f"{_tex_num(k, sp[k])} / {_tex_num(k, om[k])}")
        L.append(" & ".join(cells) + r" \\")
        if o == objects[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 2: SPIDER-CEM tracking + health + foot-slip (per object mean).
    keys2 = TRACK_KEYS + HEALTH_KEYS
    L += [r"\begin{table}[t]", r"\centering",
          r"\caption{SPIDER-CEM motion tracking, health, and foot-slide (per-object mean).}",
          r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{l r " + " ".join("r" for _ in keys2) + r"}", r"\toprule"]
    L.append(" & ".join(["Object", "n"] + [METRIC_BY_KEY[k]["label"] for k in keys2]) + r" \\")
    L.append(r"\midrule")
    for o in objects + ["ALL"]:
        ocases = cases if o == "ALL" else [c for c in cases if object_of(c) == o]
        sp = summarize(spider_m, keys2, ocases)
        cells = [o, str(len(ocases))] + [_tex_num(k, sp[k]) for k in keys2]
        L.append(" & ".join(cells) + r" \\")
        if o == objects[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    TEX.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------------------
# xlsx
# --------------------------------------------------------------------------------------

def write_xlsx(records, cases, spider_m, omni_m) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    objects = [o for o in OBJECT_ORDER if any(object_of(c) == o for c in cases)]
    wb = Workbook()
    hdr = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="2F75B5")

    def style_header(ws, ncol):
        for cell in ws[1][:ncol]:
            cell.font = hdr
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Overview
    ws = wb.active
    ws.title = "Overview"
    ws.append(["metric", "SPIDER-CEM mean", "SPIDER-CEM std", "SPIDER-CEM worst", "SPIDER n",
               "OmniRetarget mean", "OmniRetarget std", "OmniRetarget worst", "Omni n"])
    sp_all = summarize(spider_m, SPIDER_KEYS, cases)
    om_all = summarize(omni_m, OMNI_KEYS, cases)
    for k in SPIDER_KEYS:
        s = sp_all[k]
        o = om_all.get(k)
        row = [header_label(k), _num(s["mean"]), _num(s["std"]), _num(s["worst"]), s["n"]]
        if o:
            row += [_num(o["mean"]), _num(o["std"]), _num(o["worst"]), o["n"]]
        else:
            row += ["", "", "", ""]
        ws.append(row)
    style_header(ws, 9)

    # ByObject
    ws = wb.create_sheet("ByObject")
    head = ["object", "method", "n"] + [header_label(k) for k in SPIDER_KEYS]
    ws.append(head)
    for o in objects:
        ocases = [c for c in cases if object_of(c) == o]
        sp = summarize(spider_m, SPIDER_KEYS, ocases)
        ws.append([o, "SPIDER-CEM", len(ocases)] + [_num(sp[k]["mean"]) for k in SPIDER_KEYS])
        om = summarize(omni_m, OMNI_KEYS, ocases)
        ws.append([o, "OmniRetarget", len(ocases)] +
                  [_num(om[k]["mean"]) if k in om else "" for k in SPIDER_KEYS])
    style_header(ws, len(head))

    # ByCase
    ws = wb.create_sheet("ByCase")
    head = ["case_id", "object", "method", "status"] + [header_label(k) for k in SPIDER_KEYS]
    ws.append(head)
    for r in records:
        m = r.get("metrics", {})
        ws.append([r["case_id"], r["object_key"], r["method"], r["status"]] +
                  [_num(m.get(k)) for k in SPIDER_KEYS])
    style_header(ws, len(head))

    # Provenance
    ws = wb.create_sheet("Provenance")
    ws.append(["case_id", "method", "status", "source_exp", "cem_npz / omnirt_raw_npz",
               "provenance(per-metric)", "xcheck_abs", "notes"])
    for r in records:
        src = r.get("cem_npz", "") or r.get("omnirt_raw_npz", "")
        ws.append([r["case_id"], r["method"], r["status"], r.get("source_exp", ""), src,
                   json.dumps(r.get("provenance", {}), ensure_ascii=False),
                   json.dumps({k: round(v, 5) for k, v in r.get("xcheck_abs", {}).items()}),
                   ";".join(r.get("notes", []))])
    style_header(ws, 8)

    for w in wb.worksheets:
        w.freeze_panes = "A2"
    wb.save(XLSX)


def _num(v: Any) -> Any:
    return float(v) if _finite(v) else ""


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------

def write_provenance(records, cases) -> None:
    xchecks = []
    for r in records:
        for k, v in r.get("xcheck_abs", {}).items():
            if _finite(v) and float(v) > 1e-3:
                xchecks.append({"case_id": r["case_id"], "method": r["method"], "metric": k, "abs_delta": v})
    payload = {
        "n_cases": len(cases),
        "unresolved_spider": [r["case_id"] for r in records
                              if r["method"] == "SPIDER-CEM" and r["status"] != RESOLVED],
        "unresolved_omni": [r["case_id"] for r in records
                            if r["method"] == "OmniRetarget" and r["status"] != RESOLVED],
        "xcheck_mismatches_gt_1e-3": xchecks,
        "records": records,
    }
    PROV.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------------------

def write_all_reports(records: list[dict], cases: list[str]) -> None:
    spider_m = method_metric(records, "SPIDER-CEM")
    omni_m = method_metric(records, "OmniRetarget")
    write_by_case_tsv(records)
    write_markdown(records, cases, spider_m, omni_m)
    write_latex(cases, spider_m, omni_m)
    write_xlsx(records, cases, spider_m, omni_m)
    write_provenance(records, cases)
    print(f"wrote:\n  {BY_CASE_TSV}\n  {MD}\n  {XLSX}\n  {TEX}\n  {PROV}")
