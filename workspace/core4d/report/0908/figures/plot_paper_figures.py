"""Generate paper figures for the CORE4D SDF-contact / part-wise-penetration report.

Reproduces the exact numerics of the final E167A+PRG+G1 line:
  - contact_hdmi:  gain=5.0, sigma=0.30 m   (coarse, anchor-distance attraction)
  - surface_band:  scale=1.5, sigma=1.5 mm, band=[-1mm,+3mm], mode=symmetric_abs (fine, SDF)
  - robot/leg/hand-floor penalty: hinge  -scale*max(0, margin - sdf), scale=2, margin=2cm
  - hand gate: min_sdf=-10mm, hard_floor=-20mm, max_violation_pct=10%
Run:  /root/miniconda3/bin/python plot_paper_figures.py
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# colour-blind-safe
C_COARSE = "#0072B2"   # blue   - HDMI coarse
C_FINE = "#D55E00"     # orange - SDF fine band
C_SUM = "#111111"      # black  - combined
C_PEN = "#CC3311"      # red    - penetration / gate reject
C_OK = "#009E73"       # green  - accepted


def contact_reward_landscape(path: str) -> None:
    """Multi-scale contact reward vs hand-to-surface signed distance.

    coarse = Euclidean anchor attraction (NOT SDF); fine = SDF surface band.
    Both applied simultaneously; the transition is distance-driven, not a schedule.
    """
    gain, sig_far = 5.0, 0.30
    scale_band, sig_near = 1.5, 0.0015
    band_lo, band_hi = -0.001, 0.003  # symmetric_abs support band [-1mm, +3mm]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # ---- (a) far / coarse regime: 0 .. 0.4 m ----
    d = np.linspace(0.0, 0.40, 800)
    r_hdmi = gain * np.exp(-np.abs(d) / sig_far)
    axL.plot(d * 100, r_hdmi, color=C_COARSE, lw=2.2,
             label=r"HDMI coarse  $g\,e^{-\|p-p^\star\|/\sigma_c}$")
    # surface band is ~0 out here (only within +/- few mm)
    axL.axvspan(band_lo * 100, band_hi * 100, color=C_FINE, alpha=0.18)
    axL.annotate("fine SDF band\n(±few mm, see right)", xy=(0.3, 0.3),
                 xytext=(6, 3.1), fontsize=9, color=C_FINE,
                 arrowprops=dict(arrowstyle="->", color=C_FINE, lw=1.2))
    axL.set_title("(a) coarse regime — long-range attraction")
    axL.set_xlabel("hand → object surface distance  [cm]")
    axL.set_ylabel("contact reward")
    axL.set_xlim(0, 40)
    axL.set_ylim(0, 5.4)
    axL.legend(loc="upper right", frameon=False, fontsize=9)

    # ---- (b) near / fine regime: -6mm .. +12mm ----
    dm = np.linspace(-0.006, 0.012, 1600)
    r_far_near = gain * np.exp(-np.abs(dm) / sig_far)  # HDMI ~flat, ~=gain here
    band_score = scale_band * np.exp(-np.abs(dm) / sig_near)
    in_band = (dm >= band_lo) & (dm <= band_hi)
    r_band = np.where(in_band, band_score, 0.0)
    r_sum = r_far_near + r_band

    axR.plot(dm * 1000, r_far_near, color=C_COARSE, lw=1.8, ls="--",
             label=r"HDMI coarse (saturated $\approx g$)")
    axR.plot(dm * 1000, r_band, color=C_FINE, lw=2.2,
             label=r"SDF fine band  $s\,e^{-|d|/\sigma_f}$")
    axR.plot(dm * 1000, r_sum, color=C_SUM, lw=1.4, alpha=0.8,
             label="combined")
    # penetration region
    axR.axvspan(-6, 0, color=C_PEN, alpha=0.10)
    axR.axvline(0.0, color="#888888", lw=1.0, ls=":")
    axR.text(-3.0, 6.2, "penetration\n(d<0)\n→ penalty + gate",
             color=C_PEN, fontsize=8.5, ha="center", va="top")
    # band edges
    for x in (band_lo, band_hi):
        axR.axvline(x * 1000, color=C_FINE, lw=0.9, ls=":")
    axR.text(1.0, 6.7, "support band\n[-1, +3] mm", color=C_FINE,
             fontsize=8.5, ha="center")
    axR.set_title("(b) fine regime — surface lock-in")
    axR.set_xlabel("signed distance  $d=\\mathrm{SDF}$  [mm]")
    axR.set_ylabel("contact reward")
    axR.set_xlim(-6, 12)
    axR.set_ylim(0, 7.2)
    axR.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.suptitle("Multi-Scale Contact Reward "
                 "(coarse Euclidean anchor attraction $+$ fine SDF surface adherence, "
                 "applied simultaneously)",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def penetration_penalty_gate(path: str) -> None:
    """Part-wise penetration: soft hinge reward (left) + per-part hard gate (right)."""
    fig, (axP, axG) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # ---- (a) soft hinge penalty: -w*max(0, margin - sdf), w=2, margin=2cm ----
    sdf = np.linspace(-0.03, 0.05, 900)
    scale, margin = 2.0, 0.02
    pen = -scale * np.clip(margin - sdf, 0.0, None)
    axP.plot(sdf * 100, pen, color=C_COARSE, lw=2.4,
             label=r"$-w\,\max(0,\,\delta-\mathrm{SDF})$,  $w{=}2,\ \delta{=}2$cm")
    axP.axvspan(-3, 0, color=C_PEN, alpha=0.10)
    axP.axvline(0.0, color="#888888", lw=1.0, ls=":")
    axP.axvline(2.0, color="#888888", lw=0.8, ls=":")
    axP.annotate("margin $\\delta$", xy=(2.0, -0.005), xytext=(3.2, -0.02),
                 fontsize=9, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
    axP.text(-1.5, -0.093, "penetration\n(SDF<0)", color=C_PEN, fontsize=9, ha="center")
    axP.text(3.4, 0.004, "no penalty\n(clear of surface)", color="#555555",
             fontsize=8.5, ha="center")
    axP.set_title(r"(a) soft penalty per part group  (identical form)")
    axP.set_xlabel("part → object SDF  [cm]")
    axP.set_ylabel("penalty added to reward")
    axP.set_xlim(-3, 5)
    axP.set_ylim(-0.105, 0.02)
    axP.legend(loc="lower right", frameon=False, fontsize=9)

    # ---- (b) per-part HARD gate thresholds (they differ per part) ----
    # final-line: leg min_sdf=+5mm ; body(safety) min_sdf=-5mm ; hand min_sdf=-10mm, floor=-20mm
    parts = [
        ("leg  (PRG-G)", 5.0, None, C_OK),
        ("body (safety)", -5.0, None, C_COARSE),
        ("hand", -10.0, -20.0, C_FINE),
    ]
    axG.axvspan(-25, 15, color="#f2f2f2", alpha=0.0)  # keep frame
    for i, (name, thr, floor, col) in enumerate(parts):
        y = i
        # accept region to the right of threshold
        axG.plot([thr, 14], [y, y], color=col, lw=6, alpha=0.35, solid_capstyle="butt")
        axG.plot([thr], [y], marker="|", ms=18, color=col, mew=2.5)
        axG.text(14.4, y, "accept →", color=col, fontsize=8.5, va="center")
        lbl = f"{name}: min-SDF ≥ {thr:.0f} mm"
        if floor is not None:
            axG.plot([floor], [y], marker="x", ms=8, color=C_PEN, mew=2.0)
            axG.plot([floor, thr], [y, y], color=C_PEN, lw=1.4, ls=":")
            lbl += f"  (hard floor {floor:.0f} mm)"
        axG.text(-24, y + 0.28, lbl, color=col, fontsize=8.6, va="bottom")
    axG.axvline(0.0, color="#888888", lw=1.0, ls=":")
    axG.text(0.4, 2.62, "surface (SDF=0)", color="#888888", fontsize=8, rotation=90, va="top")
    axG.set_title("(b) hard candidate gate — thresholds differ per part")
    axG.set_xlabel("candidate min-SDF over rollout  [mm]")
    axG.set_yticks([0, 1, 2])
    axG.set_yticklabels(["leg", "body", "hand"])
    axG.set_ylim(-0.5, 3.0)
    axG.set_xlim(-25, 15)
    axG.spines["left"].set_visible(False)
    axG.tick_params(left=False)

    fig.suptitle(r"Part-wise Penetration Constraints: one per-part constraint "
                 r"$\varphi_P \geq \theta_P$ enforced at two levels" "\n"
                 "(a) soft differentiable penalty  $+$  (b) hard candidate gate "
                 "(per-part thresholds, AND-combined, least-violation fallback)",
                 y=1.05, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    contact_reward_landscape(os.path.join(here, "contact_reward_landscape.png"))
    penetration_penalty_gate(os.path.join(here, "penetration_penalty_gate.png"))
    print("wrote figures to", here)
