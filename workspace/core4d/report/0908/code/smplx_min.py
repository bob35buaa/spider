"""Minimal NumPy SMPLX forward (LBS) -- no `smplx`/torch dependency.

Enough to re-pose a CORE4D SMPLX sequence with a chosen shape. The CORE4D pose
npz stores per-frame vertices baked with the subject's own (often very lean)
betas; re-running the forward with `betas=0` yields the neutral average body
while keeping the exact pose and world placement.

Only the arrays present in SMPLX_NEUTRAL.npz are used:
  v_template, shapedirs, posedirs, J_regressor, weights, kintree_table,
  hands_componentsl/r, hands_meanl/r.
"""
from __future__ import annotations

import numpy as np


class SMPLXModel:
    """Holds the static SMPLX arrays and runs the LBS forward."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.v_template = np.asarray(d["v_template"], np.float64)          # (V,3)
        self.shapedirs = np.asarray(d["shapedirs"], np.float64)            # (V,3,400)
        self.posedirs = np.asarray(d["posedirs"], np.float64)              # (V,3,486)
        self.J_regressor = np.asarray(d["J_regressor"], np.float64)        # (55,V)
        self.weights = np.asarray(d["weights"], np.float64)                # (V,55)
        self.parents = np.asarray(d["kintree_table"], np.int64)[0].copy()  # (55,)
        self.parents[0] = -1
        self.faces = np.asarray(d["f"], np.int64)
        self.hands_comp_l = np.asarray(d["hands_componentsl"], np.float64)  # (45,45)
        self.hands_comp_r = np.asarray(d["hands_componentsr"], np.float64)
        self.hands_mean_l = np.asarray(d["hands_meanl"], np.float64)        # (45,)
        self.hands_mean_r = np.asarray(d["hands_meanr"], np.float64)
        self.n_joints = self.parents.shape[0]                               # 55

    def _hand_aa(self, pca: np.ndarray, comp: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """PCA hand coeffs -> (15,3) axis-angle (mean-added, use_pca convention)."""
        n = pca.shape[0]
        aa = mean + pca @ comp[:n]        # (45,)
        return aa.reshape(15, 3)

    def full_pose(self, frame: dict) -> np.ndarray:
        """Assemble the (55,3) axis-angle pose from a CORE4D frame dict."""
        go = np.asarray(frame["global_orient"], np.float64).reshape(1, 3)
        body = np.asarray(frame["body_pose"], np.float64).reshape(21, 3)
        head = np.zeros((3, 3))           # jaw, left-eye, right-eye
        lh = self._hand_aa(np.asarray(frame["left_hand_pose"], np.float64),
                           self.hands_comp_l, self.hands_mean_l)
        rh = self._hand_aa(np.asarray(frame["right_hand_pose"], np.float64),
                           self.hands_comp_r, self.hands_mean_r)
        return np.concatenate([go, body, head, lh, rh], axis=0)            # (55,3)

    def forward(self, frame: dict, betas: np.ndarray | None = None,
                return_joints: bool = False):
        """Return posed vertices (V,3) in the CORE4D world frame.

        betas=None -> neutral average shape (zeros). Pass the frame's own betas
        to reproduce the baked vertices (used for validation). With
        return_joints, also returns the (55,3) posed joints (pelvis = index 0)."""
        b = np.zeros(10) if betas is None else np.asarray(betas, np.float64).reshape(-1)[:10]
        v_shaped = self.v_template + np.einsum("vij,j->vi", self.shapedirs[..., :b.shape[0]], b)
        J = self.J_regressor @ v_shaped                                    # (55,3)

        pose = self.full_pose(frame)
        R = _rodrigues(pose)                                               # (55,3,3)
        pose_feat = (R[1:] - np.eye(3)[None]).reshape(-1)                  # (486,)
        v_posed = v_shaped + np.einsum("vip,p->vi", self.posedirs, pose_feat)

        A, posed_J = _rigid_transform_chain(R, J, self.parents)            # (55,4,4),(55,3)
        T = np.einsum("vj,jmn->vmn", self.weights, A)                      # (V,4,4)
        v_homo = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
        verts = np.einsum("vmn,vn->vm", T, v_homo)[:, :3]
        transl = np.asarray(frame["transl"], np.float64).reshape(1, 3)
        if return_joints:
            return verts + transl, posed_J + transl
        return verts + transl


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    n = rv.shape[0]
    theta = np.linalg.norm(rv, axis=1, keepdims=True)                      # (N,1)
    r = rv / np.clip(theta, 1e-8, None)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -r[:, 2], r[:, 1]
    K[:, 1, 0], K[:, 1, 2] = r[:, 2], -r[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -r[:, 1], r[:, 0]
    ct = np.cos(theta).reshape(n, 1, 1)
    st = np.sin(theta).reshape(n, 1, 1)
    return np.eye(3)[None] + st * K + (1.0 - ct) * (K @ K)


def _rigid_transform_chain(R: np.ndarray, J: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Per-joint LBS transform A[i] (55,4,4) that maps rest verts to posed."""
    nj = R.shape[0]
    rel = J.copy()
    rel[1:] -= J[parents[1:]]
    mats = np.zeros((nj, 4, 4))
    mats[:, :3, :3] = R
    mats[:, :3, 3] = rel
    mats[:, 3, 3] = 1.0
    chain = [mats[0]]
    for i in range(1, nj):
        chain.append(chain[parents[i]] @ mats[i])
    T = np.stack(chain, axis=0)                                            # (55,4,4) global
    posed_joints = T[:, :3, 3].copy()                                      # (55,3)
    Jh = np.concatenate([J, np.zeros((nj, 1))], axis=1)[:, :, None]        # (55,4,1)
    init = (T @ Jh)[:, :, 0]                                               # (55,4)
    A = T.copy()
    A[:, :, 3] -= init
    return A, posed_joints
