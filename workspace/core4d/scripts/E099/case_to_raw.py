"""E099 case_name → raw CORE4D (date, seq, person, object_name) 映射。

来源:
- d003_box021_<date>_<seq>_p<n> / e091_*<date>_*_<seq>_p<n>: 直接 parse case_name
- box021/023/025_person*: 来自 workspace/core4d_collab_retarget/results/E016/contact_masks/*/audit_summary_3cm.json
- box026_person2: base_template = box021_person1 但 object swapped 为 box026；
  raw mocap 对应 box021 18030 但物体几何不对，标记 SKIP_FINGERTIP
- box022_*: pending，目前 manifest 标 pending_e099_or_e102；按 case_name parse date/seq；
  raw 数据在 CORE4D 里有 (20231022) 但 spider 仓库还没处理；E099 best-effort
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CORE4D_ROOT = Path(
    "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real/human_object_motions"
)


@dataclass
class RawSpec:
    case: str
    date: str
    seq: str
    person: str  # "person1" or "person2"
    obj_name: str  # "Box021" / "Box023" / ...
    note: str = ""  # SKIP_FINGERTIP / MISSING_RAW / OK

    @property
    def person_npz(self) -> Path:
        return CORE4D_ROOT / self.date / self.seq / f"{self.person}_poses.npz"

    @property
    def obj_poses_npy(self) -> Path:
        return CORE4D_ROOT / self.date / self.seq / "smooth_objposes.npy"

    @property
    def metadata_json(self) -> Path:
        return CORE4D_ROOT / self.date / self.seq / "object_metadata.json"

    def raw_ok(self) -> bool:
        return self.person_npz.is_file() and self.obj_poses_npy.is_file()


# 老 case 映射 (从 E016 audit_summary_3cm.json 抽出)
LEGACY_MAP = {
    "box021_person1": ("20231018", "030", "person1", "Box021"),
    # box021_person2 已经 deprecated; 用 d003_box021_20231018_030_p1 取代
    "box023_person1": ("20231008", "045", "person1", "Box023"),
    "box023_person2": ("20231008", "045", "person2", "Box023"),
    "box025_person1": ("20231011", "048", "person1", "Box025"),
    "box025_person2": ("20231011", "048", "person2", "Box025"),
}


def parse_case(case: str) -> RawSpec:
    """case_name → RawSpec."""
    if case in LEGACY_MAP:
        date, seq, person, obj = LEGACY_MAP[case]
        return RawSpec(case=case, date=date, seq=seq, person=person, obj_name=obj)

    if case == "box026_person2":
        # base_template box021_person1 (20231018/030 person1)，物体 swap 为 box026
        # raw fingertip 对应 box021 几何，face vote 无意义
        return RawSpec(
            case=case, date="20231018", seq="030", person="person1",
            obj_name="Box026", note="SKIP_FINGERTIP_GEOMETRY_MISMATCH",
        )

    # d003_box021_<date>_<seq>_p<n>: 8 chars date, 3 chars seq
    # e091_box004_<date>_<digit>_<seq>_p<n> 形如 e091_box004_20231003_2_083_p1
    # e091_box021/box026_<date>_<seq>_p<n>
    # box022_<date>_<seq>_p<n>
    parts = case.split("_")
    person_part = parts[-1]
    assert person_part.startswith("p"), f"unexpected case: {case}"
    person_idx = int(person_part[1:])
    person = f"person{person_idx}"

    # date 是 yyyymmdd, 8 chars 全数字
    date_idx = next((i for i, p in enumerate(parts) if len(p) == 8 and p.isdigit()), None)
    assert date_idx is not None, f"no date in case: {case}"
    date = parts[date_idx]
    # seq 是紧跟 date 的下一个数字段；如 e091_box004_20231003_2_083_p1 → 中间 '2' 是子目录
    # 看 CORE4D 目录结构：20231003_1, 20231003_2 都是 date 级别
    # 但 case_name 里写法是 "e091_box004_20231003_2_083_p1" → 实际 raw 路径是 20231003_2/083/
    # box021 18029 → 20231018/029/，无子目录
    next_part = parts[date_idx + 1]
    if len(next_part) == 1 and next_part.isdigit():
        # 子目录变体
        date_full = f"{date}_{next_part}"
        seq = parts[date_idx + 2]
    else:
        date_full = date
        seq = next_part

    # 物体名：从 case_name 找 box### / bucket### 等
    obj_name = None
    for p in parts:
        if p.startswith("box") and len(p) >= 6 and p[3:].isdigit():
            obj_name = p.replace("box", "Box")
            break
    assert obj_name is not None, f"no object in case: {case}"

    return RawSpec(case=case, date=date_full, seq=seq, person=person, obj_name=obj_name)


# 自检
if __name__ == "__main__":
    test_cases = [
        ("d003_box021_20231018_029_p2", "20231018", "029", "person2", "Box021"),
        ("d003_box021_20231011_035_p2", "20231011", "035", "person2", "Box021"),
        ("e091_box004_20231003_2_083_p1", "20231003_2", "083", "person1", "Box004"),
        ("e091_box026_20231018_039_p2", "20231018", "039", "person2", "Box026"),
        ("box023_person1", "20231008", "045", "person1", "Box023"),
        ("box021_person1", "20231018", "030", "person1", "Box021"),
        ("box022_20231022_001_p2", "20231022", "001", "person2", "Box022"),
        ("box026_person2", "20231018", "030", "person1", "Box026"),
    ]
    for case, exp_date, exp_seq, exp_person, exp_obj in test_cases:
        r = parse_case(case)
        ok = (r.date, r.seq, r.person, r.obj_name) == (exp_date, exp_seq, exp_person, exp_obj)
        marker = "✓" if ok else "✗"
        print(f"{marker} {case}: date={r.date} seq={r.seq} person={r.person} obj={r.obj_name} "
              f"raw_ok={r.raw_ok()} note={r.note}")
        if not ok:
            print(f"  EXPECTED: {exp_date}/{exp_seq}/{exp_person}/{exp_obj}")
