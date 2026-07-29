"""
Fit SMPL to generated (or ground-truth) HumanML3D motion and render a mesh.

HumanML3D's packed 6D rotations were re-derived against its own unit-length
skeleton (t2m_raw_offsets), so they cannot be fed to SMPL directly -- that is
what produces the mangled body. Instead this optimises SMPL parameters until
SMPL's own joints match the target joint positions, SMPLify-style but with
only a joint term (no image, no 2D keypoints).

    # generated
    python fit_hml.py "a person walks forward" "a person kicks with the leg"

    # ground truth, to sanity-check the fitter
    python fit_hml.py --gt 000000 000021

Two-stage fit: global (root orientation + translation + scale) first, then
the full body pose. Roughly 30-60s per clip on GPU.
"""
import os
import sys
import argparse
import time

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


# --------------------------------------------------------------- 263 -> joints
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


def recover_from_ric(data, joints_num=22):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
                     positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


# ------------------------------------------------------------------- the fitter
def fit_smpl(bm, target, device, iters_global=300, iters_full=700,
             lr_global=0.05, lr_full=0.02,
             w_smooth=8.0, w_reg=0.0008, verbose=True):
    """
    bm      : BodyModel on `device`
    target  : (T, 22, 3) tensor of joint positions to match
    returns : root_orient (T,3), pose_body (T,63), trans (T,3), scale (scalar)
    """
    T = target.shape[0]
    target = target.to(device)

    root_orient = torch.zeros(T, 3, device=device, requires_grad=True)
    pose_body = torch.zeros(T, 63, device=device, requires_grad=True)
    trans = target[:, 0].clone().detach().requires_grad_(True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)

    def smpl_joints(scale_on=True):
        out = bm(root_orient=root_orient, pose_body=pose_body, trans=trans)
        J = out.Jtr[:, :22]
        if scale_on:
            root = J[:, :1]
            J = root + (J - root) * torch.exp(log_scale)
        return J

    # --- stage 1: global alignment only -------------------------------------
    opt = torch.optim.Adam([root_orient, trans, log_scale], lr=lr_global)
    for i in range(iters_global):
        opt.zero_grad()
        J = smpl_joints()
        # root + hips + shoulders carry the global frame
        idx = [0, 1, 2, 16, 17]
        loss = ((J[:, idx] - target[:, idx]) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        if verbose and (i + 1) % 100 == 0:
            print(f'    global {i+1}/{iters_global}  loss {loss.item():.5f}  '
                  f'scale {float(torch.exp(log_scale)):.3f}')

    # --- stage 2: full pose --------------------------------------------------
    opt = torch.optim.Adam([root_orient, pose_body, trans, log_scale], lr=lr_full)
    for i in range(iters_full):
        opt.zero_grad()
        J = smpl_joints()
        data_loss = ((J - target) ** 2).sum(-1).mean()
        # temporal smoothness on pose, keeps the fit from jittering
        smooth = ((pose_body[1:] - pose_body[:-1]) ** 2).mean() \
               + ((root_orient[1:] - root_orient[:-1]) ** 2).mean()
        reg = (pose_body ** 2).mean()
        loss = data_loss + w_smooth * smooth + w_reg * reg
        loss.backward()
        opt.step()
        if verbose and (i + 1) % 200 == 0:
            mpjpe = (J - target).norm(dim=-1).mean()
            print(f'    full  {i+1}/{iters_full}  data {data_loss.item():.5f}  '
                  f'mpjpe {mpjpe.item()*1000:.1f}mm')

    with torch.no_grad():
        J = smpl_joints()
        mpjpe = (J - target).norm(dim=-1).mean().item() * 1000
    return (root_orient.detach(), pose_body.detach(), trans.detach(),
            float(torch.exp(log_scale)), mpjpe)


# ------------------------------------------------------------------- rendering
def shade(v, f):
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    L = np.array([0.35, -0.65, 0.6]); L /= np.linalg.norm(L)
    b = np.clip(n @ L, 0, 1) * 0.55 + 0.45
    return t, np.clip(b[:, None] * np.array([0.89, 0.78, 0.68]), 0, 1)


def save_mesh_gif(V, faces, title, path, fps=12, elev=12, azim=-60, pad=0.35, step=2):
    V = V[::step][:, :, [0, 2, 1]].copy()          # Y-up -> Z-up
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
    ap.add_argument('items', nargs='+')
    ap.add_argument('--gt', action='store_true')
    ap.add_argument('--ckpt', default='./exps/humanml3d-263/checkpoint_0100.pth.tar')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--bm', default=os.path.expanduser('~/HumanML3D/body_models/smplh/neutral/model.npz'))
    ap.add_argument('--outdir', default='./fit_hml')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num_frames', type=int, default=60)
    ap.add_argument('--iters_global', type=int, default=300)
    ap.add_argument('--iters_full', type=int, default=700)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    std = np.load(os.path.join(args.datapath, 'Std.npy'))
    std[std < 1e-6] = 1.0

    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(bm_fname=args.bm, num_betas=10).to(dev)
    faces = bm.f.detach().cpu().numpy().astype(np.int32)

    # ---- gather (T, 263) sequences ----
    seqs, titles = [], []
    if args.gt:
        for cid in args.items:
            p = os.path.join(args.datapath, 'new_joint_vecs', f'{cid}.npy')
            if not os.path.exists(p):
                print(f'{cid}: not found'); continue
            m = np.load(p)[:args.num_frames]
            seqs.append(m); titles.append(f'GT {cid}')
    else:
        import clip
        sys.argv = ['x', args.ckpt, '--input_file', 'assets/paper_texts.txt']
        from src.parser.visualize import parser as vparser
        from src.utils.get_model_and_data import get_model_and_data
        from src.utils.misc import load_model_wo_clip

        params, folder, ckpt_name, epoch = vparser()
        params.update({'device': dev, 'dataset': 'humanml3d',
                       'datapath': args.datapath, 'pose_rep': 'xyz',
                       'num_frames': args.num_frames})

        print('loading model...')
        model, _ = get_model_and_data(params, split='vald')
        load_model_wo_clip(model, torch.load(args.ckpt, map_location=dev))
        model.eval()

        tok = clip.tokenize(args.items, truncate=True).to(dev)
        feat = model.clip_model.encode_text(tok).float().unsqueeze(0)
        dur = torch.ones((len(args.items), 1), dtype=int) * params['num_frames']
        with torch.no_grad():
            g = model.generate(feat, dur, is_amass=False, is_clip_features=True)

        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1) * std + mean
        for i, prompt in enumerate(args.items):
            seqs.append(out[i]); titles.append(prompt)

    # ---- fit and render ----
    for seq, title in zip(seqs, titles):
        print(f'\n=== {title} ===')
        d = torch.from_numpy(np.asarray(seq)).float()
        target = recover_from_ric(d, 22)
        print(f'  target joints {tuple(target.shape)}  '
              f'root xz travel {np.round(np.ptp(target[:,0,[0,2]].numpy(), axis=0), 2)}')

        t0 = time.time()
        root_orient, pose_body, trans, scale, mpjpe = fit_smpl(
            bm, target, dev, args.iters_global, args.iters_full)
        print(f'  fitted in {time.time()-t0:.1f}s   final MPJPE {mpjpe:.1f}mm   scale {scale:.3f}')

        with torch.no_grad():
            V = []
            for s in range(0, len(d), 128):
                V.append(bm(root_orient=root_orient[s:s+128],
                            pose_body=pose_body[s:s+128],
                            trans=trans[s:s+128]).v.cpu().numpy())
        V = np.concatenate(V, 0)

        safe = ''.join(c if c.isalnum() else '_' for c in title)[:45].strip('_')
        path = os.path.join(args.outdir, f'{safe}.gif')
        save_mesh_gif(V, faces, f'{title}  ({mpjpe:.0f}mm)', path)
        print(f'  -> {path}')

    print(f'\ndone -> {args.outdir}/')


if __name__ == '__main__':
    main()
