"""
Generate the same prompt in several languages and render them side by side,
so the divergence is visible rather than just a number.

    python compare_langs.py
    python compare_langs.py --frames 140 --actions walk kick

One GIF per action: five panels in a row, English first, each labelled with
its drift from the English motion.
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

from common.quaternion import qrot, qinv


GROUPS = {
    "walk": {
        "en": "a person walks forward",
        "fr": "une personne marche vers l'avant",
        "es": "una persona camina hacia adelante",
        "pt": "uma pessoa caminha para frente",
        "ru": "человек идёт вперёд",
        "zh": "一个人向前走",
        "ar": "شخص يمشي إلى الأمام",
    },
    "kick": {
        "en": "a person kicks with the leg",
        "fr": "une personne donne un coup de pied",
        "es": "una persona da una patada con la pierna",
        "pt": "uma pessoa chuta com a perna",
        "ru": "человек бьёт ногой",
        "zh": "一个人用腿踢",
        "ar": "شخص يركل بقدمه",
    },
    "jump": {
        "en": "a person jumps",
        "fr": "une personne saute",
        "es": "una persona salta",
        "pt": "uma pessoa pula",
        "ru": "человек прыгает",
        "zh": "一个人跳跃",
        "ar": "شخص يقفز",
    },
    "run": {
        "en": "a person is running",
        "fr": "une personne court",
        "es": "una persona está corriendo",
        "pt": "uma pessoa está correndo",
        "ru": "человек бежит",
        "zh": "一个人在跑步",
        "ar": "شخص يجري",
    },
    "sit": {
        "en": "a person sits down",
        "fr": "une personne s'assoit",
        "es": "una persona se sienta",
        "pt": "uma pessoa se senta",
        "ru": "человек садится",
        "zh": "一个人坐下",
        "ar": "شخص يجلس",
    },
}

# English label per language, since the script titles are in latin script
LANG_NAME = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ar": "Arabic",
}


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    ang = torch.zeros_like(rot_vel)
    ang[..., 1:] = rot_vel[..., :-1]
    ang = torch.cumsum(ang, dim=-1)
    quat = torch.zeros(data.shape[:-1] + (4,))
    quat[..., 0] = torch.cos(ang)
    quat[..., 2] = torch.sin(ang)
    pos = torch.zeros(data.shape[:-1] + (3,))
    pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    pos = qrot(qinv(quat), pos)
    pos = torch.cumsum(pos, dim=-2)
    pos[..., 1] = data[..., 3]
    return quat, pos


def recover_from_ric(data, joints_num=22):
    quat, pos = recover_root_rot_pos(data)
    p = data[..., 4:(joints_num - 1) * 3 + 4]
    p = p.view(p.shape[:-1] + (-1, 3))
    p = qrot(qinv(quat[..., None, :]).expand(p.shape[:-1] + (4,)), p)
    p[..., 0] += pos[..., 0:1]
    p[..., 2] += pos[..., 2:3]
    return torch.cat([pos.unsqueeze(-2), p], dim=-2)


CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ['orange', 'green', 'purple', 'blue', 'red']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actions', nargs='*', default=['walk', 'kick', 'jump', 'run', 'sit'])
    ap.add_argument('--ckpt', default='./exps/vae-multilingual-mclip/checkpoint_0020.pth.tar')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--outdir', default='./lang_compare')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--elev', type=float, default=12)
    ap.add_argument('--azim', type=float, default=-60)
    ap.add_argument('--zip', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sys.argv = ['x', args.ckpt, '--input_file', 'assets/paper_texts.txt']
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    params, folder, ckpt_name, epoch = vparser()
    params.update({'device': args.device, 'dataset': 'humanml3d',
                   'datapath': args.datapath, 'pose_rep': 'xyz',
                   'text_encoder': 'mclip'})

    print('loading model...')
    model, _ = get_model_and_data(params, split='vald')
    load_model_wo_clip(model, torch.load(args.ckpt, map_location=args.device))
    model.eval()

    mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    std = np.load(os.path.join(args.datapath, 'Std.npy'))
    std[std < 1e-6] = 1.0

    for action in args.actions:
        if action not in GROUPS:
            print(f'unknown action "{action}", skipping')
            continue
        group = GROUPS[action]
        langs = list(group.keys())

        with torch.no_grad():
            feat = model.clip_model.encode_text_raw(
                [group[l] for l in langs]).float().unsqueeze(0).to(args.device)
            dur = torch.ones((len(langs), 1), dtype=int) * args.frames
            g = model.generate(feat, dur, is_amass=False, is_clip_features=True)

        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1) * std + mean
        del g
        torch.cuda.empty_cache()

        motions, drifts = [], []
        for i in range(len(langs)):
            j = recover_from_ric(torch.from_numpy(out[i]).float()).numpy()
            motions.append(j[::args.step][:, :, [0, 2, 1]])   # Y-up -> Z-up
            drifts.append(float(np.abs(out[i] - out[0]).mean()))

        T = min(m.shape[0] for m in motions)
        motions = [m[:T] for m in motions]

        # one shared box so panels are directly comparable
        allpts = np.concatenate([m.reshape(-1, 3) for m in motions])
        lo, hi = allpts.min(0) - 0.4, allpts.max(0) + 0.4
        mid = (hi + lo) / 2
        rng = max((hi - lo).max() / 2, 1.0)

        n = len(langs)
        fig = plt.figure(figsize=(3.4 * n, 4.2))
        axes = [fig.add_subplot(1, n, i + 1, projection='3d') for i in range(n)]
        fig.suptitle(f'"{group["en"]}"  —  same semantic prompt, {n} languages',
                     fontsize=12)

        def update(f):
            for ax, m, l, d in zip(axes, motions, langs, drifts):
                ax.clear()
                p = m[f]
                for ch, c in zip(CHAIN, COLORS):
                    ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], c=c, linewidth=2.5)
                tr = m[:f + 1, 0]
                ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), mid[2] - rng),
                        c='0.65', lw=1.2)
                ax.set_xlim(mid[0] - rng, mid[0] + rng)
                ax.set_ylim(mid[1] - rng, mid[1] + rng)
                ax.set_zlim(mid[2] - rng, mid[2] + rng)
                ax.set_box_aspect([1, 1, 1])
                ax.view_init(elev=args.elev, azim=args.azim)
                ax.set_axis_off()
                title = LANG_NAME[l] if l == 'en' else f'{LANG_NAME[l]}  (drift {d:.4f})'
                ax.set_title(title, fontsize=9)

        path = os.path.join(args.outdir, f'{action}_7langs.gif')
        FuncAnimation(fig, update, frames=T,
                      interval=1000 // args.fps).save(path, writer='pillow',
                                                      fps=args.fps)
        plt.close(fig)
        print(f'{action:6s} -> {path}   {T} frames   '
              f'drift {[round(d, 4) for d in drifts[1:]]}')

    if args.zip:
        import shutil
        z = shutil.make_archive(args.outdir.rstrip('/'), 'zip', args.outdir)
        print(f'\nzipped -> {z}  ({os.path.getsize(z)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
