#!/usr/bin/env python
"""E165 8case 诊断: (1) ee_body_pos 失败的手/脚归因; (2) 逐 clip 数据质量对照(验证 box021 假设)。

纯离线: 读 SUGAR 失败 rollout npz(executed body_pos_b) + motion folder(reference body_pos_w),
在 pelvis-relative 去旋转帧下逐 body 算 tracking error, 在失败帧对 {双腕,双踝} 取 argmax 归因手/脚。
数据质量从 motion folder 的 reference 轨迹本身度量(接触/抬升/被跟踪 body 速度与 jerk)。

SUGAR_BODY_ORDER(14): pelvis0, L_hip1, L_knee2, L_ankle3, R_hip4, R_knee5, R_ankle6,
  torso7, L_shoulder8, L_elbow9, L_wrist10, R_shoulder11, R_elbow12, R_wrist13
"""
import glob
import os
import re
import numpy as np

SUGAR = "/mnt/public/usr/yancilin/work_dir/embodied/SUGAR-private"
OUT = "/mnt/public/usr/yancilin/work_dir/embodied/spider/workspace/core4d/results/E165/8case_diag"
PELVIS, ANKLES, WRISTS = 0, [3, 6], [10, 13]
TRACKED = ANKLES + WRISTS  # ee_body_pos termination 监控的 4 个 body

# case -> (motion_folder, staggered_eval_dir, staggered_success)
CASES = {
    "box021_r160(029_p2)": ("Core4D_E163N_Box021_R160", "e163_refiner_rl/box021_r160/spider_e163/eval_staggered_phase_mw30", 0.672),
    "box021_035_p1":       ("Core4D_E163N8_d003_box021_20231011_035_p1", "e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p1/spider_e163/eval_staggered_phase_mw30_latest_checkpoint", 0.234),
    "box021_035_p2":       ("Core4D_E163N8_d003_box021_20231011_035_p2", "e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p2/spider_e163/eval_staggered_phase_mw30_latest_checkpoint", 0.016),
    "box004_r161(083_p2)": ("Core4D_E163N_Box004_R161", "e163_refiner_rl/box004_r161/spider_e163/eval_staggered_phase_mw30", 0.484),
    "box004_082_p1":       ("Core4D_E163N8_e091_box004_20231003_2_082_p1", "e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_082_p1/spider_e163/eval_staggered_phase_mw30_latest_checkpoint", 0.094),
    "box004_083_p1":       ("Core4D_E163N8_e091_box004_20231003_2_083_p1", "e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_083_p1/spider_e163/eval_staggered_phase_mw30_latest_checkpoint", 0.016),
    "box023_r158":         ("Core4D_E163N_Box023_R158", "e163_refiner_rl/box023_r158/spider_e163/eval_staggered_phase_mw30", 0.000),
}


def quat_rotate_inverse(q, v):
    """IsaacLab 约定 q=(w,x,y,z), 返回 R(q)^T v。q:(...,4) v:(...,3)."""
    qw = q[..., 0:1]
    qvec = q[..., 1:]
    a = v * (2.0 * qw**2 - 1.0)
    b = np.cross(qvec, v) * qw * 2.0
    c = qvec * (np.sum(qvec * v, axis=-1, keepdims=True)) * 2.0
    return a - b + c


def body_b_from_world(body_pos_w, root_pos_w, root_quat_w):
    """每帧把 body 世界坐标转到 root(pelvis) 局部去旋转帧。"""
    rel = body_pos_w - root_pos_w[:, None, :]  # (T,14,3)
    rq = root_quat_w[:, None, :].repeat(rel.shape[1], axis=1)  # (T,14,4)
    return quat_rotate_inverse(rq, rel)


def load_ref_body_b(motion_folder):
    d = np.load(f"{SUGAR}/data/{motion_folder}/data_000/robot_50hz.npz", allow_pickle=True)
    bw, bq = d["body_pos_w"], d["body_quat_w"]
    ref_b = body_b_from_world(bw, bw[:, PELVIS], bq[:, PELVIS])
    return ref_b, d


def handfoot_attribution(name, motion_folder, eval_dir):
    ref_b, _ = load_ref_body_b(motion_folder)
    npzs = sorted(glob.glob(f"{SUGAR}/outputs/core4d/{eval_dir}/rollout/raw_npz/ee_body_pos/*.npz"))
    if not npzs:
        return None
    valid = 0
    foot_win = hand_win = 0
    body_err_accum = np.zeros(4)  # 失败帧 4 body 平均 error
    conv_check = []
    for f in npzs:
        m = re.search(r"_t(\d+)-(\d+)_", os.path.basename(f))
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        z = np.load(f, allow_pickle=True)
        # executed body 用与 reference 完全相同的 pelvis-relative 去旋转构造(保证同口径)
        exe_b = body_b_from_world(z["body_pos_w"], z["body_pos_w"][:, PELVIS], z["body_quat_w"][:, PELVIS])
        T = exe_b.shape[0]
        if e > ref_b.shape[0]:
            e = ref_b.shape[0]
        ref_win = ref_b[s:e]
        if ref_win.shape[0] != T:
            n = min(ref_win.shape[0], T)
            ref_win, exe_b = ref_win[:n], exe_b[:n]
        # 自校验: 同口径构造与 npz 自带 body_pos_b 的中位差(仅供参考, 不影响同口径归因)
        if "body_pos_b" in z.files:
            conv_check.append(float(np.median(np.linalg.norm(exe_b[: z["body_pos_b"].shape[0]] - z["body_pos_b"][: exe_b.shape[0]], axis=-1))))
        err = np.linalg.norm(exe_b - ref_win, axis=-1)  # (T,14)
        last = err[-1, TRACKED]  # 失败帧 4 个 tracked body 的 error
        body_err_accum += last
        amax = TRACKED[int(np.argmax(last))]
        if amax in WRISTS:
            hand_win += 1
        else:
            foot_win += 1
        valid += 1
    return {
        "valid": valid, "hand_win": hand_win, "foot_win": foot_win,
        "hand_share": hand_win / valid if valid else float("nan"),
        "mean_err_LR_ankle_wrist": (body_err_accum / valid).round(3).tolist() if valid else None,
        "conv_check_median_m": round(float(np.median(conv_check)), 4) if conv_check else None,
    }


def data_quality(name, motion_folder):
    ref_b, d = load_ref_body_b(motion_folder)
    bw = d["body_pos_w"]
    fps = int(d["fps"][0]) if "fps" in d.files else 50
    dt = 1.0 / fps
    T = bw.shape[0]
    # contact
    cl = np.load(f"{SUGAR}/data/{motion_folder}/data_000/contact_labels_50hz.npy")
    contact_ratio = float(np.mean(cl))
    # object
    import pickle
    with open(f"{SUGAR}/data/{motion_folder}/data_000/obj_motion_global_50hz.pkl", "rb") as fp:
        obj = pickle.load(fp)
    ot = np.asarray(obj["obj_trans"])
    obj_z_lift = float(ot[:, 2].max() - ot[:, 2].min())
    obj_speed = np.linalg.norm(np.diff(ot, axis=0), axis=-1) / dt
    # tracked body world speed/jerk (wrist+ankle)
    tb = bw[:, TRACKED]  # (T,4,3)
    vel = np.linalg.norm(np.diff(tb, axis=0), axis=-1) / dt  # (T-1,4)
    acc = np.linalg.norm(np.diff(tb, n=2, axis=0), axis=-1) / dt**2  # (T-2,4)
    jerk = np.linalg.norm(np.diff(tb, n=3, axis=0), axis=-1) / dt**3
    wrist = bw[:, WRISTS]
    wvel = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) / dt
    ankle = bw[:, ANKLES]
    avel = np.linalg.norm(np.diff(ankle, axis=0), axis=-1) / dt
    root_z = bw[:, PELVIS, 2]
    jv = d["joint_vel"] if "joint_vel" in d.files else np.diff(d["joint_pos"], axis=0) / dt
    return {
        "frames": T, "contact_ratio": round(contact_ratio, 3),
        "obj_z_lift_m": round(obj_z_lift, 3), "obj_speed_max": round(float(obj_speed.max()), 2),
        "trackbody_speed_max": round(float(vel.max()), 2),
        "wrist_speed_max": round(float(wvel.max()), 2), "ankle_speed_max": round(float(avel.max()), 2),
        "trackbody_acc_max": round(float(acc.max()), 1), "trackbody_jerk_p95": round(float(np.percentile(jerk, 95)), 0),
        "root_z_min": round(float(root_z.min()), 3),
        "joint_vel_max": round(float(np.abs(jv).max()), 2),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 100)
    P("PART 1 — ee_body_pos 失败的 手 vs 脚 归因 (失败帧 argmax over {L/R ankle, L/R wrist})")
    P("=" * 100)
    P(f"{'case':<22}{'success':>8}{'ee_fail':>8}{'hand%':>8}{'foot%':>8}  mean_err[Lank,Rank,Lwr,Rwr]  conv_chk")
    for name, (mf, ed, succ) in CASES.items():
        r = handfoot_attribution(name, mf, ed)
        if r is None:
            P(f"{name:<22}{succ:>8.3f}{'  no ee_body npz':>8}")
            continue
        P(f"{name:<22}{succ:>8.3f}{r['valid']:>8}{r['hand_share']*100:>7.0f}%{(1-r['hand_share'])*100:>7.0f}%"
          f"  {r['mean_err_LR_ankle_wrist']}  {r['conv_check_median_m']}")
    P()
    P("解读: hand% = 失败帧 4 个 tracked body 里 error 最大者是手(腕)的占比; foot% 是脚(踝)。")
    P("注: ref/exe 均用同一 pelvis-relative 去旋转构造, 归因同口径可信。")
    P("conv_chk = 同口径构造 vs npz 自带 body_pos_b 的中位差(~7cm = 存储 root 与 pelvis 的偏置, 不影响相对归因)。")

    P()
    P("=" * 100)
    P("PART 2 — 逐 clip 数据质量(全部来自 reference 轨迹本身, 按 staggered success 降序)")
    P("=" * 100)
    hdr = f"{'case':<22}{'succ':>6}{'frm':>5}{'contact':>8}{'objLift':>8}{'objV':>6}{'trkVmax':>8}{'wrVmax':>7}{'ankVmax':>8}{'accMax':>7}{'jerkP95':>9}{'rootZ':>7}{'jVmax':>7}"
    P(hdr)
    for name, (mf, ed, succ) in sorted(CASES.items(), key=lambda x: -x[1][2]):
        q = data_quality(name, mf)
        P(f"{name:<22}{succ:>6.3f}{q['frames']:>5}{q['contact_ratio']:>8}{q['obj_z_lift_m']:>8}{q['obj_speed_max']:>6}"
          f"{q['trackbody_speed_max']:>8}{q['wrist_speed_max']:>7}{q['ankle_speed_max']:>8}{q['trackbody_acc_max']:>7}"
          f"{q['trackbody_jerk_p95']:>9.0f}{q['root_z_min']:>7}{q['joint_vel_max']:>7}")
    P()
    P("列说明: contact=接触帧占比; objLift=物体 z 抬升幅度(m); objV=物体最大速度(m/s);")
    P("trkVmax/wrVmax/ankVmax=被跟踪body/腕/踝最大世界速度(m/s); accMax=被跟踪body最大加速度;")
    P("jerkP95=被跟踪body jerk 95分位; rootZ=pelvis 最低高度(m,蹲伏程度); jVmax=关节最大角速度(rad/s)。")

    with open(f"{OUT}/handfoot_dataquality.txt", "w") as fp:
        fp.write("\n".join(lines))
    print(f"\n[written] {OUT}/handfoot_dataquality.txt")


if __name__ == "__main__":
    main()
