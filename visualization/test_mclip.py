"""
Test whether MotionCLIP's latent space transfers across languages, on either
model. M-CLIP replaces CLIP's English-only text encoder; the decoder is
untouched.

    pip install multilingual-clip

    # the authors' released AMASS model
    python test_mclip.py --model paper

    # your HumanML3D 263-dim model
    python test_mclip.py --model hml

    # both, back to back
    python test_mclip.py --model both

Checks, in order:
  1. dimension     -- M-CLIP must emit 512
  2. alignment     -- cosine similarity between languages, and vs CLIP English
  3. generation    -- decode each language, render, and measure how far the
                      generated motions drift from the English one
"""
import os
import sys
import argparse

sys.path.append('.')
sys.path.append(os.path.expanduser('~/HumanML3D'))

import numpy as np
import torch
import clip as openai_clip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


GROUPS = {
    'walk': {
        'en': "a person walks forward",
        'fr': "une personne marche vers l'avant",
        'ar': "شخص يمشي إلى الأمام",
        'zh': "一个人向前走",
        'ru': "человек идёт вперёд",
    },
    'kick': {
        'en': "a person kicks with the leg",
        'fr': "une personne donne un coup de pied",
        'ar': "شخص يركل بقدمه",
        'zh': "一个人用腿踢",
        'ru': "человек бьёт ногой",
    },
    'jump': {
        'en': "a person jumps",
        'fr': "une personne saute",
        'ar': "شخص يقفز",
        'zh': "一个人跳跃",
        'ru': "человек прыгает",
    },
}

MODELS = {
    'paper': dict(ckpt='./exps/paper-model/checkpoint_0100.pth.tar',
                  datapath='./data/amass_db/amass_30fps_db.pt',
                  dataset='amass', kind='smpl'),
    'hml':   dict(ckpt='./exps/humanml3d-mclip/checkpoint_0100.pth.tar',
                  datapath='./data/HumanML3D',
                  dataset='humanml3d', kind='263'),
}


# ------------------------------------------------------- 263 -> joint positions
def _recover_root(data):
    from common.quaternion import qrot, qinv
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
    from common.quaternion import qrot, qinv
    quat, pos = _recover_root(data)
    p = data[..., 4:(joints_num - 1) * 3 + 4]
    p = p.view(p.shape[:-1] + (-1, 3))
    p = qrot(qinv(quat[..., None, :]).expand(p.shape[:-1] + (4,)), p)
    p[..., 0] += pos[..., 0:1]
    p[..., 2] += pos[..., 2:3]
    return torch.cat([pos.unsqueeze(-2), p], dim=-2)


# ------------------------------------------------------------------- rendering
CHAIN_22 = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ['orange', 'green', 'purple', 'blue', 'red']


def save_gif(joints, title, path, axes, flip, fps=20, elev=12, azim=-60, pad=0.4):
    m = joints[:, :, axes].copy()
    if flip:
        m[:, :, 2] = -m[:, :, 2]
    pts = m.reshape(-1, 3)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    mid = (hi + lo) / 2
    rng = max((hi - lo).max() / 2, 1.0)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.clear()
        p = m[i]
        for ch, c in zip(CHAIN_22, COLORS):
            ch = [j for j in ch if j < p.shape[0]]
            ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], c=c, linewidth=3)
        tr = m[:i + 1, 0]
        ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), mid[2] - rng), c='0.65', lw=1.2)
        ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng); ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=5, pad=-2)
        ax.set_title(title, fontsize=8)

    FuncAnimation(fig, update, frames=len(m),
                  interval=1000 // fps).save(path, writer='pillow', fps=fps)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def run_model(tag, cfg, mc, tk, args):
    print('\n' + '=' * 62)
    print(f'3. GENERATION -- {tag}  ({cfg["kind"]})')
    print('=' * 62)

    sys.argv = ['x', cfg['ckpt'], '--input_file', 'assets/paper_texts.txt']
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    params, folder, ckpt_name, epoch = vparser()
    params['device'] = args.device
    params['datapath'] = cfg['datapath']
    params['dataset'] = cfg['dataset']
    if cfg['kind'] == '263':
        params['pose_rep'] = 'xyz'

    model, _ = get_model_and_data(params, split='vald')
    load_model_wo_clip(model, torch.load(cfg['ckpt'], map_location=args.device))
    model.eval()

    if cfg['kind'] == '263':
        mean = np.load(os.path.join(cfg['datapath'], 'Mean.npy'))
        std = np.load(os.path.join(cfg['datapath'], 'Std.npy'))
        std[std < 1e-6] = 1.0

    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    for gname, group in GROUPS.items():
        langs = list(group.keys())
        with torch.no_grad():
            # the entire swap: M-CLIP instead of clip.encode_text
            feat = mc.forward([group[l] for l in langs], tk).float().unsqueeze(0).to(args.device)
            dur = torch.ones((len(langs), 1), dtype=int) * args.frames
            g = model.generate(feat, dur,
                               is_amass=(cfg['kind'] == 'smpl'),
                               is_clip_features=True)

            if cfg['kind'] == '263':
                out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)
                out = out * std + mean
                joints = [recover_from_ric(torch.from_numpy(o).float()).numpy()
                          for o in out]
                axes, flip = [0, 2, 1], False
            else:
                J = model.rotation2xyz(g['output'], g['mask'], pose_rep='rot6d',
                                       translation=True, glob=True,
                                       jointstype='smpl', vertstrans=True).cpu().numpy()
                joints = [np.transpose(J[i], (2, 0, 1))[:, :22] for i in range(len(langs))]
                out = np.stack([j.reshape(j.shape[0], -1) for j in joints])
                axes, flip = [0, 2, 1], True

        base = out[0]
        print(f'\n  "{group["en"]}"')
        for i, l in enumerate(langs):
            d = float(np.abs(out[i] - base).mean())
            path = os.path.join(outdir, f'{gname}_{l}.gif')
            save_gif(joints[i], f'[{tag}/{l}] {group[l][:38]}', path, axes, flip)
            travel = np.ptp(joints[i][:, 0, [0, 2]], axis=0).round(2)
            print(f'  {l:>5}  drift vs en {d:>8.4f}   travel {travel}')

    del model
    torch.cuda.empty_cache()
    print(f'\n  -> {outdir}/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='both', choices=['paper', 'hml', 'both'])
    ap.add_argument('--outdir', default='./mclip_test')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--mclip', default='M-CLIP/XLM-Roberta-Large-Vit-B-32')
    ap.add_argument('--skip_gen', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 1. dimension ----
    print('=' * 62)
    print('1. DIMENSION CHECK')
    print('=' * 62)
    from src.utils.mclip_wrapper import MClipTextEncoder

    enc = MClipTextEncoder(name=args.mclip, device=args.device)
    tk = enc.tokenizer

    class _Shim:
        def forward(self, texts, _tok=None):
            return enc.encode_text_raw(texts)
    mc = _Shim()

    with torch.no_grad():
        probe = mc.forward([GROUPS['walk']['en']], tk)
    dim = probe.shape[-1]
    print(f'  {args.mclip}')
    print(f'  output {tuple(probe.shape)} -> {dim}-dim')
    if dim != 512:
        print(f'\n  STOP: decoder expects 512, got {dim}.')
        return
    print('  OK\n')

    # ---- 2. alignment ----
    print('=' * 62)
    print('2. ALIGNMENT -- languages vs each other, and vs CLIP English')
    print('=' * 62)
    cmodel, _ = openai_clip.load('ViT-B/32', device=args.device, jit=False)

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))

    for gname, group in GROUPS.items():
        langs = list(group.keys())
        with torch.no_grad():
            embs = mc.forward([group[l] for l in langs], tk).float().cpu()
            ref = cmodel.encode_text(
                openai_clip.tokenize([group['en']]).to(args.device)).float().cpu()
        print(f'\n  "{group["en"]}"')
        print(f'  {"lang":>5}  {"vs M-CLIP en":>13}  {"vs CLIP en":>11}')
        for i, l in enumerate(langs):
            print(f'  {l:>5}  {cos(embs[i:i+1], embs[0:1]):>13.4f}  '
                  f'{cos(embs[i:i+1], ref):>11.4f}')

    print('\n  Column 1 near 1.0: all languages land in the same place.')
    print('  Column 2 near 1.0: that place is where CLIP puts the English.')
    print('  Low values are the interesting case, not a failure -- that gap')
    print('  is what a PID analysis is meant to quantify.')

    del cmodel
    torch.cuda.empty_cache()
    if args.skip_gen:
        return

    # ---- 3. generation ----
    tags = ['paper', 'hml'] if args.model == 'both' else [args.model]
    for tag in tags:
        cfg = MODELS[tag]
        if not os.path.exists(cfg['ckpt']):
            print(f'\n  skip {tag}: {cfg["ckpt"]} not found')
            continue
        try:
            run_model(tag, cfg, mc, tk, args)
        except Exception as e:
            print(f'\n  {tag} FAILED: {type(e).__name__}: {e}')

    print(f'\ndone -> {args.outdir}/')


if __name__ == '__main__':
    main()
