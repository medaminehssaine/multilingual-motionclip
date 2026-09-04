"""
Text-to-motion from the HumanML3D 263-dim MotionCLIP, with per-prompt lengths.

    # varied lengths, one per prompt
    python gen_hml.py "a person walks forward:120" "a person punches:40"

    # same length for all
    python gen_hml.py --frames 90 "a person walks forward" "a person jumps"

    # built-in set with sensible per-action durations
    python gen_hml.py --preset

gendurations is per-row, so a batch can mix lengths. Anything past the
training length (60) is extrapolation -- the decoder queries positional
encodings it never saw -- so expect drift in the tail beyond ~120.
"""
import os
import sys
import argparse

sys.path.append('.')
sys.path.append(os.path.expanduser('~/HumanML3D'))

import numpy as np
import torch
import clip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.quaternion import qrot, qinv


# (prompt, frames) -- short actions get short clips, locomotion gets long ones
PRESET = [
    ("a person walks forward",                    160),
    ("a person walks backward",                   140),
    ("a person is running",                       130),
    ("a person walks in a circle",                180),
    ("a person runs and then stops",              150),
    ("a person turns around and walks away",      160),
    ("a person jumps",                             60),
    ("a person jumps on one leg",                 100),
    ("a person kicks with the leg",                70),
    ("a person punches with the right hand",       60),
    ("a person squats down and stands up",        110),
    ("a person sits down on a chair",             100),
    ("a person stands up and walks",              140),
    ("a person raises both arms above the head",   70),
    ("a person waves the right hand",              80),
    ("a person crawls forward on the ground",     150),
    ("a person dances in place",                  170),
    ("a person steps to the side twice",          100),
    ("a person bends over and picks something up", 120),
    ("a person throws a ball",                     70),
]


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


# ------------------------------------------------------------------- rendering
CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
COLORS = ['orange', 'green', 'purple', 'blue', 'red']


def save_gif(joints, title, path, fps=20, elev=12, azim=-60, pad=0.4, step=1):
    m = joints[::step][:, :, [0, 2, 1]].copy()      # Y-up -> Z-up

    pts = m.reshape(-1, 3)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    mid = (hi + lo) / 2
    rng = max((hi - lo).max() / 2, 1.0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.clear()
        p = m[i]
        for ch, c in zip(CHAIN, COLORS):
            ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], c=c, linewidth=3)
        trail = m[:i + 1, 0]
        ax.plot(trail[:, 0], trail[:, 1],
                np.full(len(trail), mid[2] - rng), c='0.65', lw=1.2)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X', fontsize=7, labelpad=-6)
        ax.set_ylabel('Y', fontsize=7, labelpad=-6)
        ax.set_zlabel('Z', fontsize=7, labelpad=-6)
        ax.tick_params(labelsize=5, pad=-2)
        ax.set_title(f'"{title}"   {len(m)} frames @ {fps}fps', fontsize=9)

    FuncAnimation(fig, update, frames=len(m),
                  interval=1000 // fps).save(path, writer='pillow', fps=fps)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def parse_items(raw, default_frames):
    """'text:120' -> ('text', 120);  'text' -> ('text', default)"""
    out = []
    for r in raw:
        if ':' in r and r.rsplit(':', 1)[1].strip().isdigit():
            txt, n = r.rsplit(':', 1)
            out.append((txt.strip(), int(n)))
        else:
            out.append((r.strip(), default_frames))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prompts', nargs='*')
    ap.add_argument('--preset', action='store_true',
                    help='(default when no prompts are given)')
    ap.add_argument('--frames', type=int, default=60,
                    help='default length when not given per prompt')
    ap.add_argument('--ckpt', default='./exps/humanml3d-263/checkpoint_0100.pth.tar')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--outdir', default='./gen_hml')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--text_encoder', default='mclip',
                    choices=['mclip', 'perlang', 'clip'],
                    help='MUST match how the checkpoint was trained')
    ap.add_argument('--joints_num', type=int, default=22)
    ap.add_argument('--fps', type=int, default=20,
                    help='HumanML3D is 20fps; keep for real-time playback')
    ap.add_argument('--step', type=int, default=1, help='render every Nth frame')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--zip', action='store_true', default=True,
                    help='zip the output folder when done')
    ap.add_argument('--no-zip', dest='zip', action='store_false')
    args = ap.parse_args()

    if args.prompts:
        items = parse_items(args.prompts, args.frames)
    else:
        items = PRESET      # no prompts given -> run the built-in set

    os.makedirs(args.outdir, exist_ok=True)

    sys.argv = ['x', args.ckpt, '--input_file', 'assets/paper_texts.txt']
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    params, folder, ckpt_name, epoch = vparser()
    params.update({'text_encoder': args.text_encoder,
                   'device': args.device, 'dataset': 'humanml3d',
                   'datapath': args.datapath, 'pose_rep': 'xyz'})

    print('loading model...')
    model, _ = get_model_and_data(params, split='vald')
    load_model_wo_clip(model, torch.load(args.ckpt, map_location=args.device))
    model.eval()

    trained = params.get('num_frames', 60)
    longest = max(n for _, n in items)
    if longest > trained:
        print(f'note: longest request is {longest} frames, model trained on {trained}. '
              f'Positions past {trained} were never seen -- expect the tail to drift.')

    mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    std = np.load(os.path.join(args.datapath, 'Std.npy'))
    std[std < 1e-6] = 1.0

    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        texts = [t for t, _ in chunk]
        lens = [n for _, n in chunk]

        if hasattr(model.clip_model, 'encode_text_raw'):
            # M-CLIP / per-language bank take raw strings, not CLIP tokens
            feat = model.clip_model.encode_text_raw(texts).float().unsqueeze(0)
        else:
            tok = clip.tokenize(texts, truncate=True).to(args.device)
            feat = model.clip_model.encode_text(tok).float().unsqueeze(0)
        # per-row durations -- this is what lets lengths differ within a batch
        dur = torch.tensor(lens, dtype=torch.int).reshape(-1, 1)

        with torch.no_grad():
            g = model.generate(feat, dur, is_amass=False, is_clip_features=True)

        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)   # (B, T, 263)
        out = out * std + mean
        del g
        torch.cuda.empty_cache()

        for j, (prompt, n) in enumerate(chunk):
            # the batch pads to the longest row; trim back to this prompt's length
            seq = out[j][:n]
            joints = recover_from_ric(torch.from_numpy(seq).float(),
                                      args.joints_num).numpy()
            safe = ''.join(c if c.isalnum() else '_' for c in prompt)[:45].strip('_')
            path = os.path.join(args.outdir, f'{s+j:02d}_{safe}_{n}f.gif')
            save_gif(joints, prompt, path, fps=args.fps, step=args.step)
            travel = np.ptp(joints[:, 0, [0, 2]], axis=0).round(2)
            print(f'[{s+j+1}/{len(items)}] {path}   {joints.shape[0]}f   '
                  f'root xz travel {travel}')

    print(f'\ndone -> {args.outdir}/   ({len(items)} gifs)')

    if args.zip:
        import shutil
        z = shutil.make_archive(args.outdir.rstrip('/'), 'zip', args.outdir)
        mb = os.path.getsize(z) / 1e6
        print(f'zipped -> {z}   ({mb:.1f} MB)')


if __name__ == '__main__':
    main()
