"""
Evaluate MotionCLIP checkpoints with Guo et al.'s HumanML3D evaluator,
following MoMask's protocol and reporting format exactly.

    python eval_momask.py --model ml
    python eval_momask.py --model ml-en --repeat 20

Matches momask-codes/eval_t2m_trans_res.py in:
  - get_co_embeddings() for both towers, which sorts lengths internally
  - R-precision accumulated over the whole loader with sum_all=True,
    divided by nb_sample
  - matching score as euclidean_distance_matrix(et, em).trace() / nb_sample
  - diversity with 300 if nb_sample > 300 else 100
  - 20 repeats, conf = np.std(x) * 1.96 / sqrt(repeat_time)

ONE DEVIATION, and it matters for how the CI is read.

MoMask regenerates motion each repeat with a different seed, so its CI
captures sampling variance. MotionCLIP is a deterministic autoencoder:
identical text gives identical motion every time. Regenerating 20 times
would produce a CI of exactly zero.

So generation happens once, and the 20 repeats reshuffle the retrieval
grouping instead. The CI therefore reflects variance in which clips are
compared against each other, not variance in the model. Report it as such.
"""
import os
import sys
import argparse
import codecs as cs

MOMASK = os.path.expanduser('~/momask-codes')
sys.path.insert(0, MOMASK)
sys.path.append('.')
sys.path.append(os.path.expanduser('~/HumanML3D'))

import numpy as np
import torch

from utils.metrics import (calculate_R_precision, calculate_diversity,
                           calculate_activation_statistics,
                           calculate_frechet_distance,
                           euclidean_distance_matrix)


CKPTS = {
    'clip':     ('./exps/humanml3d-263/checkpoint_0100.pth.tar',  'clip'),
    'mclip':    ('./exps/humanml3d-mclip/checkpoint_0100.pth.tar', 'mclip'),
    'ml-en':    ('./exps/ml-en/checkpoint_0100.pth.tar',    'mclip'),
    'ml-latin': ('./exps/ml-latin/checkpoint_0100.pth.tar', 'mclip'),
    'ml-far':   ('./exps/ml-far/checkpoint_0100.pth.tar',   'mclip'),
    'ml-all':   ('./exps/ml-all/checkpoint_0100.pth.tar',   'mclip'),
    'vae-en-actor': ('./exps/vae-en-actor/checkpoint_0100.pth.tar', 'mclip'),
    'vae-en-kl1e-3': ('./exps/vae-en-kl1e-3/checkpoint_0100.pth.tar', 'mclip'),
    'vae-en-kl1e-5': ('./exps/vae-en-kl1e-5/checkpoint_0100.pth.tar', 'mclip'),
    'perlang-latin': ('./exps/perlang-latin/checkpoint_0100.pth.tar', 'perlang'),
}
GROUPS = {'ml': ['ml-en', 'ml-latin', 'ml-far', 'ml-all', 'perlang-latin'],
          'both': ['clip', 'mclip'],
          'all': list(CKPTS),
          'vae': [k for k in CKPTS if k.startswith('vae-')]}

EVAL_META = os.path.join(MOMASK, 'checkpoints/t2m/Comp_v6_KLD005/meta')


class EvalOpt:
    def __init__(self, device):
        self.device = device
        self.dataset_name = 't2m'
        self.checkpoints_dir = os.path.join(MOMASK, 'checkpoints')
        self.dim_movement_enc_hidden = 512
        self.dim_movement_latent = 512
        self.unit_length = 4


def conf(x, n):
    """MoMask's interval: std * 1.96 / sqrt(n), with numpy's default ddof=0"""
    return float(np.std(x) * 1.96 / np.sqrt(n))


# ───────────────────────────────────────────── data prep
def encode_caption(w_vec, tokens, max_text_len=20):
    if len(tokens) < max_text_len:
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        tokens = ['sos/OTHER'] + tokens[:max_text_len] + ['eos/OTHER']
        sent_len = len(tokens)
    we, po = [], []
    for t in tokens:
        w, p = w_vec[t]
        we.append(w[None, :]); po.append(p[None, :])
    return np.concatenate(we, 0), np.concatenate(po, 0), sent_len


def pad_to(m, length=196):
    T = len(m)
    if T >= length:
        return m[:length], length
    return np.concatenate([m, np.zeros((length - T, m.shape[1]), m.dtype)], 0), T


def load_testset(datapath, ev_mean, ev_std, n_max):
    ids = [l.strip() for l in open(os.path.join(datapath, 'test.txt')) if l.strip()]
    ids = [c for c in ids if not c.startswith('M')]

    motions, lens, caps, toks = [], [], [], []
    for cid in ids:
        mp = os.path.join(datapath, 'new_joint_vecs', cid + '.npy')
        tp = os.path.join(datapath, 'texts', cid + '.txt')
        if not (os.path.exists(mp) and os.path.exists(tp)):
            continue
        m = np.load(mp)
        if len(m) < 40:
            continue
        with cs.open(tp) as f:
            parts = f.readline().strip().split('#')
        if len(parts) < 2 or not parts[0]:
            continue
        tk = [t for t in parts[1].split(' ') if '/' in t]
        if not tk:
            continue
        mm, L = pad_to((m - ev_mean) / ev_std)
        motions.append(mm); lens.append(L); caps.append(parts[0]); toks.append(tk)
        if len(motions) >= n_max:
            break
    return motions, lens, caps, toks


# ───────────────────────────────────────────── embedding
def co_embed(ev, w_vec, motions, lens, toks, batch=32, device='cuda:0'):
    """Same computation as get_co_embeddings, but returns both embeddings in
    INPUT order.

    get_co_embeddings sorts by motion length and returns both towers in that
    permuted order. Real motions have varying lengths so the permutation is
    non-trivial; generated motions are all the same length so it is identity.
    Comparing across the two passes then breaks the text-motion pairing and
    R-precision collapses to chance. So sort and unsort explicitly here.
    """
    E_t, E_m = [], []
    for s in range(0, len(motions), batch):
        M = torch.from_numpy(np.stack(motions[s:s+batch])).float().to(device)
        L = torch.tensor(lens[s:s+batch])
        we, po, sl = zip(*[encode_caption(w_vec, t) for t in toks[s:s+batch]])
        we = torch.from_numpy(np.stack(we)).float().to(device)
        po = torch.from_numpy(np.stack(po)).float().to(device)
        sl = torch.tensor(sl)

        with torch.no_grad():
            # motion tower, sorted by motion length
            mo = torch.argsort(L, descending=True)
            minv = torch.argsort(mo)
            mv = ev.movement_encoder(M[mo][..., :-4]).detach()
            em = ev.motion_encoder(mv, L[mo] // ev.opt.unit_length)
            em = em.cpu().numpy()[minv.numpy()]

            # text tower, sorted by caption length
            to = torch.argsort(sl, descending=True)
            tinv = torch.argsort(to)
            et = ev.text_encoder(we[to], po[to], sl[to])
            et = et.cpu().numpy()[tinv.numpy()]

        E_t.append(et); E_m.append(em)
    return np.concatenate(E_t), np.concatenate(E_m)


def metrics_once(Et, Em, gt_mu, gt_cov, rng, batch=32):
    """one repeat: reshuffle the grouping, recompute R-precision and matching"""
    n = (len(Em) // batch) * batch
    idx = rng.permutation(len(Em))[:n]

    R = np.zeros(3); match = 0.0
    for s in range(0, n, batch):
        g = idx[s:s+batch]
        R += calculate_R_precision(Et[g], Em[g], top_k=3, sum_all=True)
        match += euclidean_distance_matrix(Et[g], Em[g]).trace()
    R /= n; match /= n

    mu, cov = calculate_activation_statistics(Em[idx])
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    div = calculate_diversity(Em[idx], 300 if n > 300 else 100)
    return float(fid), float(div), R, float(match)


def report(name, fid, div, t1, t2, t3, match, n):
    return (f'{name} final result:\n'
            f'\tFID: {np.mean(fid):.3f}, conf. {conf(fid, n):.3f}\n'
            f'\tDiversity: {np.mean(div):.3f}, conf. {conf(div, n):.3f}\n'
            f'\tTOP1: {np.mean(t1):.3f}, conf. {conf(t1, n):.3f}, '
            f'TOP2. {np.mean(t2):.3f}, conf. {conf(t2, n):.3f}, '
            f'TOP3. {np.mean(t3):.3f}, conf. {conf(t3, n):.3f}\n'
            f'\tMatching: {np.mean(match):.3f}, conf. {conf(match, n):.3f}\n')


# ───────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ml')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--out', default='eval_results.log')
    args = ap.parse_args()

    dev = args.device
    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer

    print('loading evaluator...')
    ev = EvaluatorModelWrapper(EvalOpt(dev))
    w_vec = WordVectorizer(os.path.join(MOMASK, 'glove'), 'our_vab')

    ev_mean = np.load(os.path.join(EVAL_META, 'mean.npy'))
    ev_std = np.load(os.path.join(EVAL_META, 'std.npy'))
    ds_mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    ds_std = np.load(os.path.join(args.datapath, 'Std.npy'))
    ds_std[ds_std < 1e-6] = 1.0

    motions, lens, caps, toks = load_testset(args.datapath, ev_mean, ev_std, args.n)
    print(f'{len(motions)} test clips\n')

    Et, Em = co_embed(ev, w_vec, motions, lens, toks, device=dev)
    gt_mu, gt_cov = calculate_activation_statistics(Em)

    f = open(args.out, 'w')

    # ---- ground truth ----
    fid, div, t1, t2, t3, match = [], [], [], [], [], []
    for i in range(args.repeat):
        rng = np.random.default_rng(i)
        a, b, R, m = metrics_once(Et, Em, gt_mu, gt_cov, rng)
        fid.append(a); div.append(b)
        t1.append(R[0]); t2.append(R[1]); t3.append(R[2]); match.append(m)
    msg = report('ground truth', fid, div, t1, t2, t3, match, args.repeat)
    print(msg); print(msg, file=f, flush=True)

    # ---- models ----
    tags = GROUPS.get(args.model, [args.model])
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    for tag in tags:
        if tag not in CKPTS:
            print(f'unknown model {tag}'); continue
        ckpt, enc = CKPTS[tag]
        if not os.path.exists(ckpt):
            print(f'skip {tag}: {ckpt} missing'); continue

        sys.argv = ['x', ckpt, '--input_file', 'assets/paper_texts.txt']
        params, folder, cname, epoch = vparser()
        params.update({'device': dev, 'dataset': 'humanml3d',
                       'datapath': args.datapath, 'pose_rep': 'xyz',
                       'text_encoder': enc})
        model, _ = get_model_and_data(params, split='vald')
        load_model_wo_clip(model, torch.load(ckpt, map_location=dev))
        model.eval()

        # deterministic: generate once
        gen, glens = [], []
        for s in range(0, len(caps), 16):
            chunk = caps[s:s+16]
            with torch.no_grad():
                ft = model.clip_model.encode_text_raw(chunk).float().unsqueeze(0).to(dev)
                d = torch.ones((len(chunk), 1), dtype=int) * args.frames
                g = model.generate(ft, d, is_amass=False, is_clip_features=True)
            out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)
            out = out * ds_std + ds_mean          # undo dataset norm
            for o in out:
                mm, L = pad_to((o - ev_mean) / ev_std)   # into evaluator space
                gen.append(mm); glens.append(L)
            del g; torch.cuda.empty_cache()

        _, Eg = co_embed(ev, w_vec, gen, glens, toks, device=dev)

        fid, div, t1, t2, t3, match = [], [], [], [], [], []
        for i in range(args.repeat):
            rng = np.random.default_rng(i)
            a, b, R, m = metrics_once(Et, Eg, gt_mu, gt_cov, rng)
            fid.append(a); div.append(b)
            t1.append(R[0]); t2.append(R[1]); t3.append(R[2]); match.append(m)
        msg = report(tag, fid, div, t1, t2, t3, match, args.repeat)
        print(msg); print(msg, file=f, flush=True)

        del model; torch.cuda.empty_cache()

    f.close()
    print(f'written to {args.out}')
    print('\nNote: MotionCLIP is deterministic, so the intervals above reflect')
    print('variance in retrieval grouping, not sampling variance as in MoMask.')


if __name__ == '__main__':
    main()
