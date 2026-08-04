"""
20 complex prompts x 5 languages, rendered side by side.

Optionally runs the CLIP-trained model on the same prompts for comparison --
but note CLIP's text encoder is English-only, so its non-English panels are
the control, not a fair competitor. That contrast is the point.

    python compare20.py                 # mclip only
    python compare20.py --both          # mclip + clip baseline
    python compare20.py --limit 4       # quick test on 4 prompts

Output: one GIF per prompt per model, plus a summary table and a zip.
"""
import os
import re
import sys
import json
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


LANG_NAME = {'en': 'English', 'fr': 'French', 'ar': 'Arabic',
             'zh': 'Chinese', 'ru': 'Russian'}

# 20 prompts, each in 5 languages. Deliberately varied: locomotion, in-place
# actions, sequences, manner adverbs, direction words.
PROMPTS = [
    ("walk_slow", 140, {
        'en': "a person walks forward slowly",
        'fr': "une personne marche lentement vers l'avant",
        'ar': "شخص يمشي إلى الأمام ببطء",
        'zh': "一个人缓慢地向前走",
        'ru': "человек медленно идёт вперёд"}),
    ("walk_circle", 160, {
        'en': "a person walks in a circle",
        'fr': "une personne marche en cercle",
        'ar': "شخص يمشي في دائرة",
        'zh': "一个人绕圈走",
        'ru': "человек идёт по кругу"}),
    ("run_stop", 140, {
        'en': "a person runs and then stops",
        'fr': "une personne court puis s'arrête",
        'ar': "شخص يجري ثم يتوقف",
        'zh': "一个人跑步然后停下",
        'ru': "человек бежит и затем останавливается"}),
    ("walk_back", 130, {
        'en': "a person walks backward",
        'fr': "une personne marche à reculons",
        'ar': "شخص يمشي إلى الخلف",
        'zh': "一个人向后走",
        'ru': "человек идёт назад"}),
    ("turn_walk", 150, {
        'en': "a person turns around and walks away",
        'fr': "une personne se retourne et s'éloigne",
        'ar': "شخص يستدير ويبتعد",
        'zh': "一个人转身走开",
        'ru': "человек разворачивается и уходит"}),
    ("jump_high", 70, {
        'en': "a person jumps high",
        'fr': "une personne saute haut",
        'ar': "شخص يقفز عالياً",
        'zh': "一个人高高跳起",
        'ru': "человек высоко прыгает"}),
    ("jump_one_leg", 100, {
        'en': "a person hops on one leg",
        'fr': "une personne saute à cloche-pied",
        'ar': "شخص يقفز على قدم واحدة",
        'zh': "一个人单腿跳",
        'ru': "человек прыгает на одной ноге"}),
    ("kick_high", 70, {
        'en': "a person kicks high with the right leg",
        'fr': "une personne donne un coup de pied haut avec la jambe droite",
        'ar': "شخص يركل عالياً بساقه اليمنى",
        'zh': "一个人用右腿高踢",
        'ru': "человек высоко бьёт правой ногой"}),
    ("punch_twice", 70, {
        'en': "a person punches twice with the fists",
        'fr': "une personne donne deux coups de poing",
        'ar': "شخص يلكم مرتين بقبضتيه",
        'zh': "一个人用拳头打两下",
        'ru': "человек дважды бьёт кулаками"}),
    ("squat_stand", 110, {
        'en': "a person squats down and stands back up",
        'fr': "une personne s'accroupit puis se relève",
        'ar': "شخص يجلس القرفصاء ثم يقف",
        'zh': "一个人蹲下然后站起来",
        'ru': "человек приседает и снова встаёт"}),
    ("sit_chair", 100, {
        'en': "a person sits down on a chair",
        'fr': "une personne s'assoit sur une chaise",
        'ar': "شخص يجلس على كرسي",
        'zh': "一个人坐在椅子上",
        'ru': "человек садится на стул"}),
    ("stand_walk", 130, {
        'en': "a person stands up and starts walking",
        'fr': "une personne se lève et commence à marcher",
        'ar': "شخص يقف ويبدأ بالمشي",
        'zh': "一个人站起来开始走路",
        'ru': "человек встаёт и начинает идти"}),
    ("raise_arms", 70, {
        'en': "a person raises both arms above the head",
        'fr': "une personne lève les deux bras au-dessus de la tête",
        'ar': "شخص يرفع ذراعيه فوق رأسه",
        'zh': "一个人把双臂举过头顶",
        'ru': "человек поднимает обе руки над головой"}),
    ("wave_hand", 80, {
        'en': "a person waves the right hand",
        'fr': "une personne fait signe de la main droite",
        'ar': "شخص يلوّح بيده اليمنى",
        'zh': "一个人挥右手",
        'ru': "человек машет правой рукой"}),
    ("crawl", 140, {
        'en': "a person crawls forward on the ground",
        'fr': "une personne rampe vers l'avant sur le sol",
        'ar': "شخص يزحف إلى الأمام على الأرض",
        'zh': "一个人在地上向前爬",
        'ru': "человек ползёт вперёд по земле"}),
    ("dance", 160, {
        'en': "a person dances in place",
        'fr': "une personne danse sur place",
        'ar': "شخص يرقص في مكانه",
        'zh': "一个人原地跳舞",
        'ru': "человек танцует на месте"}),
    ("step_side", 100, {
        'en': "a person steps to the side twice",
        'fr': "une personne fait deux pas de côté",
        'ar': "شخص يخطو جانباً مرتين",
        'zh': "一个人向旁边走两步",
        'ru': "человек делает два шага в сторону"}),
    ("bend_pick", 110, {
        'en': "a person bends over and picks something up",
        'fr': "une personne se penche et ramasse quelque chose",
        'ar': "شخص ينحني ويلتقط شيئاً",
        'zh': "一个人弯腰捡东西",
        'ru': "человек наклоняется и что-то поднимает"}),
    ("throw_ball", 80, {
        'en': "a person throws a ball forward",
        'fr': "une personne lance une balle vers l'avant",
        'ar': "شخص يرمي كرة إلى الأمام",
        'zh': "一个人向前扔球",
        'ru': "человек бросает мяч вперёд"}),
    ("stretch_arms", 90, {
        'en': "a person stretches both arms out to the sides",
        'fr': "une personne étire les deux bras sur les côtés",
        'ar': "شخص يمد ذراعيه إلى الجانبين",
        'zh': "一个人把双臂向两侧伸展",
        'ru': "человек вытягивает обе руки в стороны"}),
]

MODELS = {
    'mclip': dict(ckpt='./exps/humanml3d-mclip/checkpoint_0100.pth.tar',
                  text_encoder='mclip',
                  label='M-CLIP (multilingual)'),
    'clip':  dict(ckpt='./exps/humanml3d-263/checkpoint_0100.pth.tar',
                  text_encoder='clip',
                  label='CLIP (English-only)'),
}


# --------------------------------------------------------------- 263 -> joints
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


def render_row(motions, langs, drifts, title, subtitle, path, fps, step,
               elev, azim):
    motions = [m[::step] for m in motions]
    T = min(m.shape[0] for m in motions)
    motions = [m[:T] for m in motions]

    allpts = np.concatenate([m.reshape(-1, 3) for m in motions])
    lo, hi = allpts.min(0) - 0.4, allpts.max(0) + 0.4
    mid = (hi + lo) / 2
    rng = max((hi - lo).max() / 2, 1.0)

    n = len(langs)
    fig = plt.figure(figsize=(3.3 * n, 4.4))
    axes = [fig.add_subplot(1, n, i + 1, projection='3d') for i in range(n)]
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.text(0.5, 0.925, subtitle, ha='center', fontsize=9, color='0.35')

    def update(f):
        for ax, m, l, d in zip(axes, motions, langs, drifts):
            ax.clear()
            p = m[f]
            for ch, c in zip(CHAIN, COLORS):
                ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], c=c, linewidth=2.5)
            tr = m[:f + 1, 0]
            ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), mid[2] - rng),
                    c='0.65', lw=1.2)
            ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng)
            ax.set_zlim(mid[2]-rng, mid[2]+rng); ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            lab = LANG_NAME[l] if l == 'en' else f'{LANG_NAME[l]}   drift {d:.4f}'
            ax.set_title(lab, fontsize=9)

    FuncAnimation(fig, update, frames=T,
                  interval=1000 // fps).save(path, writer='pillow', fps=fps)
    plt.close(fig)
    return T


def run(tag, cfg, prompts, args):
    print(f'\n{"="*66}\n{cfg["label"]}\n{"="*66}')

    sys.argv = ['x', cfg['ckpt'], '--input_file', 'assets/paper_texts.txt']
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    params, folder, ckpt_name, epoch = vparser()
    params.update({'device': args.device, 'dataset': 'humanml3d',
                   'datapath': args.datapath, 'pose_rep': 'xyz',
                   'text_encoder': cfg['text_encoder']})

    model, _ = get_model_and_data(params, split='vald')
    load_model_wo_clip(model, torch.load(cfg['ckpt'], map_location=args.device))
    model.eval()

    mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    std = np.load(os.path.join(args.datapath, 'Std.npy'))
    std[std < 1e-6] = 1.0

    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for i, (name, nframes, group) in enumerate(prompts):
        langs = list(group.keys())
        with torch.no_grad():
            feat = model.clip_model.encode_text_raw(
                [group[l] for l in langs]).float().unsqueeze(0).to(args.device)
            dur = torch.ones((len(langs), 1), dtype=int) * nframes
            g = model.generate(feat, dur, is_amass=False, is_clip_features=True)

        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1) * std + mean
        del g
        torch.cuda.empty_cache()

        motions, drifts, travels = [], [], []
        for k in range(len(langs)):
            j = recover_from_ric(torch.from_numpy(out[k][:nframes]).float()).numpy()
            motions.append(j[:, :, [0, 2, 1]])                 # Y-up -> Z-up
            drifts.append(float(np.abs(out[k] - out[0]).mean()))
            travels.append(np.ptp(j[:, 0, [0, 2]], axis=0).round(2).tolist())

        path = os.path.join(outdir, f'{i:02d}_{name}.gif')
        render_row(motions, langs, drifts,
                   f'"{group["en"]}"',
                   f'{cfg["label"]}   ·   {nframes} frames',
                   path, args.fps, args.step, args.elev, args.azim)

        rows.append(dict(name=name, prompt=group['en'], frames=nframes,
                         drift={l: round(d, 4) for l, d in zip(langs, drifts)},
                         travel={l: t for l, t in zip(langs, travels)}))
        print(f'[{i+1:2d}/{len(prompts)}] {name:14s} '
              f'drift ' + '  '.join(f'{l}={d:.4f}' for l, d in
                                    zip(langs[1:], drifts[1:])))

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--both', action='store_true', help='also run the CLIP model')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--outdir', default='./compare20')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--elev', type=float, default=12)
    ap.add_argument('--azim', type=float, default=-60)
    args = ap.parse_args()

    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    os.makedirs(args.outdir, exist_ok=True)

    tags = ['mclip', 'clip'] if args.both else ['mclip']
    results = {}
    for tag in tags:
        cfg = MODELS[tag]
        if not os.path.exists(cfg['ckpt']):
            print(f'skip {tag}: {cfg["ckpt"]} not found')
            continue
        try:
            results[tag] = run(tag, cfg, prompts, args)
        except Exception as e:
            print(f'{tag} FAILED: {type(e).__name__}: {e}')

    # ---- summary ----
    print(f'\n{"="*66}\nMEAN DRIFT FROM ENGLISH\n{"="*66}')
    print(f'{"model":<10} ' + ''.join(f'{LANG_NAME[l]:>10}' for l in
                                      ['fr', 'ar', 'zh', 'ru']))
    for tag, rows in results.items():
        means = {l: np.mean([r['drift'][l] for r in rows])
                 for l in ['fr', 'ar', 'zh', 'ru']}
        print(f'{tag:<10} ' + ''.join(f'{means[l]:>10.4f}' for l in
                                      ['fr', 'ar', 'zh', 'ru']))

    if 'clip' in results and 'mclip' in results:
        print('\nCLIP was trained with an English-only text encoder, so its')
        print('non-English rows are a control rather than a competitor --')
        print('a large gap there is the expected result, not a surprise.')

    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nwrote {args.outdir}/results.json')

    import shutil
    z = shutil.make_archive(args.outdir.rstrip('/'), 'zip', args.outdir)
    print(f'zipped -> {z}  ({os.path.getsize(z)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
