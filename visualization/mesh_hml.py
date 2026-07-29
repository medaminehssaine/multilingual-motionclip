"""
Render HumanML3D 263-dim motion as an SMPL mesh.

The 263 vector packs 21 joints x 6D rotations at dims 67:193, and HumanML3D
uses SMPL's first 22 joints -- so those rotations can be converted back to
SMPL pose parameters and pushed through the body model to get a real mesh.

IMPORTANT: run with --gt first. If a ground-truth clip renders as a correct
human, the rotation conversion is right and generated output can be trusted.
If GT looks broken, the conversion is wrong, not the model.

    # validate on ground truth
    python mesh_hml.py --gt 000000 000021

    # then generate
    python mesh_hml.py "a person walks forward" "a person kicks"
"""
import os
import sys
import argparse

sys.path.append('.')
sys.path.append(os.path.expanduser('~/HumanML3D'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common.quaternion import qrot, qinv


# --------------------------------------------------------------- root recovery
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


# ------------------------------------------------------ 263 -> SMPL parameters
def cont6d_to_matrix(c6d):
    """(..., 6) -> (..., 3, 3). Gram-Schmidt, as in Zhou et al. 2019."""
    a1, a2 = c6d[..., :3], c6d[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_axis_angle(R):
    """(..., 3, 3) -> (..., 3), via quaternion."""
    from src.utils.rotation_conversions import matrix_to_quaternion, quaternion_to_axis_angle
    return quaternion_to_axis_angle(matrix_to_quaternion(R))


def quat_to_axis_angle(q):
    from src.utils.rotation_conversions import quaternion_to_axis_angle
    return quaternion_to_axis_angle(q)


def hml_to_smpl(data, joints_num=22):
    """data: (T, 263) unnormalised. Returns root_orient (T,3), pose_body (T,63), trans (T,3)."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    start = 4 + (joints_num - 1) * 3
    end = start + (joints_num - 1) * 6
    c6d = data[..., start:end].reshape(data.shape[0], joints_num - 1, 6)

    R = cont6d_to_matrix(c6d)                       # (T, 21, 3, 3)
    pose_body = matrix_to_axis_angle(R).reshape(data.shape[0], -1)   # (T, 63)
    root_orient = quat_to_axis_angle(r_rot_quat)                     # (T, 3)
    return root_orient, pose_body, r_pos


# ------------------------------------------------------------------- rendering
def shade(v, f):
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    L = np.array([0.35, -0.65, 0.6]); L /= np.linalg.norm(L)
    b = np.clip(n @ L, 0, 1) * 0.55 + 0.45
    return t, np.clip(b[:, None] * np.array([0.89, 0.78, 0.68]), 0, 1)


def save_mesh_gif(V, faces, title, path, fps=12, elev=12, azim=-60, pad=0.35, step=2):
    V = V[::step]
    V = V[:, :, [0, 2, 1]].copy()          # Y-up -> Z-up

    pts = V.reshape(-1, 3)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    mid = (hi + lo) / 2
    rng = max((hi - lo).max() / 2, 1.0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.clear()
        tris, cols = shade(V[i], faces)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols,
                                             edgecolors='none', linewidths=0, shade=False))
        c = V[:i + 1].mean(1)
        ax.plot(c[:, 0], c[:, 1], np.full(len(c), mid[2] - rng), c='0.65', lw=1.2)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f'{title}  [{i+1}/{len(V)}]', fontsize=9)

    FuncAnimation(fig, update, frames=len(V),
                  interval=1000 // fps).save(path, writer='pillow', fps=fps)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('items', nargs='+', help='prompts, or clip ids when --gt')
    ap.add_argument('--gt', action='store_true', help='render ground-truth clips instead')
    ap.add_argument('--ckpt', default='./exps/humanml3d-263/checkpoint_0100.pth.tar')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--bm', default=os.path.expanduser('~/HumanML3D/body_models/smplh/neutral/model.npz'))
    ap.add_argument('--outdir', default='./mesh_hml')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num_frames', type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    std = np.load(os.path.join(args.datapath, 'Std.npy'))
    std[std < 1e-6] = 1.0

    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(bm_fname=args.bm, num_betas=10).to('cpu')
    faces = bm.f.detach().cpu().numpy().astype(np.int32)

    # ---- collect (T, 263) sequences, unnormalised ----
    seqs, titles = [], []

    if args.gt:
        for cid in args.items:
            p = os.path.join(args.datapath, 'new_joint_vecs', f'{cid}.npy')
            if not os.path.exists(p):
                print(f'{cid}: not found'); continue
            seqs.append(np.load(p))                     # already unnormalised on disk
            titles.append(f'GT {cid}')
    else:
        import clip
        sys.argv = ['x', args.ckpt, '--input_file', 'assets/paper_texts.txt']
        from src.parser.visualize import parser as vparser
        from src.utils.get_model_and_data import get_model_and_data
        from src.utils.misc import load_model_wo_clip

        params, folder, ckpt_name, epoch = vparser()
        params['device'] = args.device
        params['dataset'] = 'humanml3d'
        params['datapath'] = args.datapath
        params['pose_rep'] = 'xyz'
        params['num_frames'] = args.num_frames

        print('loading model...')
        model, _ = get_model_and_data(params, split='vald')
        load_model_wo_clip(model, torch.load(args.ckpt, map_location=args.device))
        model.eval()

        tok = clip.tokenize(args.items, truncate=True).to(args.device)
        feat = model.clip_model.encode_text(tok).float().unsqueeze(0)
        dur = torch.ones((len(args.items), 1), dtype=int) * params['num_frames']
        with torch.no_grad():
            g = model.generate(feat, dur, is_amass=False, is_clip_features=True)

        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)   # (B, T, 263)
        out = out * std + mean
        for i, prompt in enumerate(args.items):
            seqs.append(out[i])
            titles.append(prompt)

    # ---- 263 -> SMPL -> mesh ----
    for seq, title in zip(seqs, titles):
        d = torch.from_numpy(seq).float()
        root_orient, pose_body, trans = hml_to_smpl(d)

        print(f'{title}: T={len(d)}  root_orient{tuple(root_orient.shape)} '
              f'pose_body{tuple(pose_body.shape)} trans travel '
              f'{np.round(np.ptp(trans.numpy(), axis=0), 2)}')

        with torch.no_grad():
            V = []
            for s in range(0, len(d), 128):
                V.append(bm(root_orient=root_orient[s:s+128],
                            pose_body=pose_body[s:s+128],
                            trans=trans[s:s+128]).v.numpy())
        V = np.concatenate(V, 0)

        safe = ''.join(c if c.isalnum() else '_' for c in title)[:45].strip('_')
        path = os.path.join(args.outdir, f'{safe}.gif')
        save_mesh_gif(V, faces, title, path)
        print(f'  -> {path}')

    print(f'\ndone -> {args.outdir}/')


if __name__ == '__main__':
    main()
