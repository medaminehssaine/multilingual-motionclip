"""
Generate GIFs from the three retrained models, each in two variants:

    fixed   pelvis pinned at origin  (the paper's behaviour, --no-vertstrans)
    moving  root translation applied

    python gen_all_models.py            # all three
    python gen_all_models.py --only mclip

Produces one zip per model:
    ~/gifs_amass.zip     my-paper-model, AMASS/BABEL, English
    ~/gifs_hml.zip       humanml3d-263, English
    ~/gifs_mclip.zip     humanml3d-mclip, 9 languages

The two 263-dim models cannot use the repo's own text2motion script -- it
assumes SMPL rotations. They are rendered here from recover_from_ric instead,
and "fixed" means the root is subtracted per frame rather than a training flag.
"""
import os
import re
import sys
import shutil
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


PROMPTS = [
    ("walk",        "a person walks forward",                  120),
    ("walk_back",   "a person walks backward",                 110),
    ("run",         "a person is running",                     110),
    ("circle",      "a person walks in a circle",              140),
    ("turn",        "a person turns around",                    80),
    ("jump",        "a person jumps",                           60),
    ("hop",         "a person hops on one leg",                 90),
    ("kick",        "a person kicks with the leg",              70),
    ("punch",       "a person punches",                         60),
    ("squat",       "a person squats down",                     80),
    ("sit",         "a person sits down",                       90),
    ("stand",       "a person stands up",                       80),
    ("raise_arms",  "a person raises both arms",                70),
    ("wave",        "a person waves the hand",                  70),
    ("crawl",       "a person crawls forward",                 120),
    ("dance",       "a person dances",                         140),
]

# for the M-CLIP model: same three actions in nine languages
MULTILANG = {
    "walk": {'en': "a person walks forward",
             'fr': "une personne marche vers l'avant",
             'es': "una persona camina hacia adelante",
             'de': "eine Person geht vorwärts",
             'ru': "человек идёт вперёд",
             'ar': "شخص يمشي إلى الأمام",
             'zh': "一个人向前走",
             'ja': "人が前に歩く",
             'hi': "एक व्यक्ति आगे चलता है"},
    "kick": {'en': "a person kicks with the leg",
             'fr': "une personne donne un coup de pied",
             'es': "una persona da una patada",
             'de': "eine Person tritt mit dem Bein",
             'ru': "человек бьёт ногой",
             'ar': "شخص يركل بقدمه",
             'zh': "一个人用腿踢",
             'ja': "人が足で蹴る",
             'hi': "एक व्यक्ति पैर से लात मारता है"},
    "jump": {'en': "a person jumps",
             'fr': "une personne saute",
             'es': "una persona salta",
             'de': "eine Person springt",
             'ru': "человек прыгает",
             'ar': "شخص يقفز",
             'zh': "一个人跳跃",
             'ja': "人がジャンプする",
             'hi': "एक व्यक्ति कूदता है"},
}
LANG_NAME = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
             'ru': 'Russian', 'ar': 'Arabic', 'zh': 'Chinese',
             'ja': 'Japanese', 'hi': 'Hindi'}

MODELS = {
    'amass': dict(ckpt='./exps/my-paper-model/checkpoint_0100.pth.tar',
                  dataset='amass', datapath='./data/amass_db/amass_30fps_db.pt',
                  pose_rep='rot6d', kind='smpl', enc='clip',
                  label='my retrain · AMASS · CLIP',
                  axes=[0, 2, 1], flip=True),
    'hml':   dict(ckpt='./exps/humanml3d-263/checkpoint_0100.pth.tar',
                  dataset='humanml3d', datapath='./data/HumanML3D',
                  pose_rep='xyz', kind='263', enc='clip',
                  label='HumanML3D · 263 · CLIP',
                  axes=[0, 2, 1], flip=False),
    'mclip': dict(ckpt='./exps/humanml3d-mclip/checkpoint_0100.pth.tar',
                  dataset='humanml3d', datapath='./data/HumanML3D',
                  pose_rep='xyz', kind='263', enc='mclip',
                  label='HumanML3D · 263 · M-CLIP',
                  axes=[0, 2, 1], flip=False),
}


# ─────────────────────────────────────────────── 263 -> joints
def _root(data):
    rv = data[..., 0]
    ang = torch.zeros_like(rv); ang[..., 1:] = rv[..., :-1]
    ang = torch.cumsum(ang, dim=-1)
    q = torch.zeros(data.shape[:-1] + (4,))
    q[..., 0] = torch.cos(ang); q[..., 2] = torch.sin(ang)
    p = torch.zeros(data.shape[:-1] + (3,))
    p[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    p = torch.cumsum(qrot(qinv(q), p), dim=-2)
    p[..., 1] = data[..., 3]
    return q, p


def recover_from_ric(data, J=22):
    q, pos = _root(data)
    p = data[..., 4:(J - 1) * 3 + 4].view(data.shape[:-1] + (-1, 3))
    p = qrot(qinv(q[..., None, :]).expand(p.shape[:-1] + (4,)), p)
    p[..., 0] += pos[..., 0:1]; p[..., 2] += pos[..., 2:3]
    return torch.cat([pos.unsqueeze(-2), p], dim=-2)


CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ['orange', 'green', 'purple', 'blue', 'red']


def save_gif(m, title, path, fps=20, step=2, elev=12, azim=-60, pad=0.4,
             trail=True):
    m = m[::step]
    pts = m.reshape(-1, 3)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    mid = (hi + lo) / 2
    rng = max((hi - lo).max() / 2, 1.0)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    def upd(i):
        ax.clear()
        p = m[i]
        for ch, c in zip(CHAIN, COLORS):
            ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], c=c, linewidth=2.6)
        if trail:
            tr = m[:i + 1, 0]
            ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), mid[2] - rng),
                    c='0.65', lw=1.2)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)

    FuncAnimation(fig, upd, frames=len(m),
                  interval=1000 // fps).save(path, writer='pillow', fps=fps)
    plt.close(fig)


def orient(j, axes, flip):
    j = j[:, :, axes].copy()
    if flip:
        j[:, :, 2] = -j[:, :, 2]
    return j


def safe(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')[:40]


# ─────────────────────────────────────────────── generation
def build(tag, cfg, device):
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    sys.argv = ['x', cfg['ckpt'], '--input_file', 'assets/paper_texts.txt']
    params, folder, cname, epoch = vparser()
    params.update({'device': device, 'dataset': cfg['dataset'],
                   'datapath': cfg['datapath'], 'pose_rep': cfg['pose_rep'],
                   'text_encoder': cfg['enc']})
    m, _ = get_model_and_data(params, split='vald')
    load_model_wo_clip(m, torch.load(cfg['ckpt'], map_location=device))
    m.eval()
    return m, params


def generate(model, cfg, texts, frames, device, mean=None, std=None):
    """returns list of (T, 22, 3), already oriented, WITH translation"""
    with torch.no_grad():
        f = model.clip_model.encode_text_raw(texts).float().unsqueeze(0).to(device)
        d = torch.ones((len(texts), 1), dtype=int) * frames
        g = model.generate(f, d, is_amass=(cfg['kind'] == 'smpl'),
                           is_clip_features=True)

        if cfg['kind'] == '263':
            out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)
            out = out * std + mean
            js = [recover_from_ric(torch.from_numpy(o).float()).numpy()
                  for o in out]
        else:
            J = model.rotation2xyz(g['output'], g['mask'], pose_rep='rot6d',
                                   translation=True, glob=True,
                                   jointstype='smpl',
                                   vertstrans=True).cpu().numpy()
            js = [np.transpose(J[i], (2, 0, 1))[:, :22] for i in range(len(texts))]
    del g
    torch.cuda.empty_cache()
    return [orient(j, cfg['axes'], cfg['flip']) for j in js]


def pin_root(j):
    """subtract the root position each frame -> pelvis fixed at origin"""
    return j - j[:, 0:1, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=list(MODELS))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outroot', default=os.path.expanduser('~'))
    args = ap.parse_args()

    MEAN = np.load('./data/HumanML3D/Mean.npy')
    STD = np.load('./data/HumanML3D/Std.npy'); STD[STD < 1e-6] = 1.0

    for tag in args.only:
        cfg = MODELS[tag]
        if not os.path.exists(cfg['ckpt']):
            print(f'skip {tag}: {cfg["ckpt"]} missing')
            continue

        print(f'\n{"="*60}\n{cfg["label"]}\n{"="*60}')
        model, params = build(tag, cfg, args.device)

        base = os.path.join(args.outroot, f'gifs_{tag}')
        shutil.rmtree(base, ignore_errors=True)
        for v in ('moving', 'fixed'):
            os.makedirs(os.path.join(base, v), exist_ok=True)

        # ---- English prompt set ----
        for i, (name, prompt, frames) in enumerate(PROMPTS):
            js = generate(model, cfg, [prompt], frames, args.device, MEAN, STD)
            j = js[0]
            travel = np.ptp(j[:, 0, :2], axis=0).round(2)

            save_gif(j, f'{prompt}', 
                     os.path.join(base, 'moving', f'{i:02d}_{name}.gif'))
            save_gif(pin_root(j), f'{prompt}  (root pinned)',
                     os.path.join(base, 'fixed', f'{i:02d}_{name}.gif'),
                     trail=False)
            print(f'  [{i+1:2d}/{len(PROMPTS)}] {name:12s} travel {travel}')

        # ---- multilingual, M-CLIP only ----
        if cfg['enc'] == 'mclip':
            for v in ('moving', 'fixed'):
                os.makedirs(os.path.join(base, v, 'multilingual'), exist_ok=True)
            for act, group in MULTILANG.items():
                codes = list(group)
                js = generate(model, cfg, [group[c] for c in codes], 100,
                              args.device, MEAN, STD)
                for c, j in zip(codes, js):
                    ttl = f'[{LANG_NAME[c]}] {act}'
                    save_gif(j, ttl, os.path.join(
                        base, 'moving', 'multilingual', f'{act}_{c}.gif'))
                    save_gif(pin_root(j), ttl + '  (root pinned)',
                             os.path.join(base, 'fixed', 'multilingual',
                                          f'{act}_{c}.gif'), trail=False)
                print(f'  multilingual: {act} x {len(codes)} languages')

        del model
        torch.cuda.empty_cache()

        z = shutil.make_archive(base, 'zip', base)
        n = sum(len(f) for _, _, f in os.walk(base))
        print(f'  -> {z}   {n} gifs   {os.path.getsize(z)/1e6:.1f} MB')

    print('\ndone')


if __name__ == '__main__':
    main()
