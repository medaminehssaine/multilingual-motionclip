"""
Evaluate MotionCLIP generations with Guo et al.'s pretrained HumanML3D
evaluator -- the same instrument every published text-to-motion number uses.

    python eval_guo.py --model mclip
    python eval_guo.py --model both --langs

Requires (from momask-codes):
    ~/momask-codes/checkpoints/t2m/text_mot_match/model/finest.tar
    ~/momask-codes/checkpoints/t2m/Comp_v6_KLD005/meta/{mean,std}.npy
    ~/momask-codes/glove/our_vab_{data.npy,words.pkl,idx.pkl}

Three things that silently ruin these numbers if you get them wrong, all
handled below:

  1. The evaluator has its OWN normalisation stats, different from the
     dataset's Mean.npy/Std.npy. Motion must be de-normalised with the dataset
     stats and re-normalised with the evaluator's.
  2. Motion is padded to exactly 196 frames, and m_lens must be divisible by
     unit_length (4) -- the movement encoder downsamples.
  3. Captions go through GloVe as word/POS tokens, capped at 20 words.

R-precision is ENGLISH ONLY. The evaluator's text tower is a GloVe BiGRU
trained on English HumanML3D captions -- it cannot read French or Chinese.
FID works for every language, since it only looks at motion.
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
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────── evaluator setup
class EvalOpt:
    def __init__(self, device):
        self.device = device
        self.dataset_name = 't2m'
        self.checkpoints_dir = os.path.join(MOMASK, 'checkpoints')
        self.dim_movement_enc_hidden = 512
        self.dim_movement_latent = 512
        self.unit_length = 4


def load_evaluator(device):
    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    return EvaluatorModelWrapper(EvalOpt(device))


def load_word_vectorizer():
    from utils.word_vectorizer import WordVectorizer
    return WordVectorizer(os.path.join(MOMASK, 'glove'), 'our_vab')


EVAL_META = os.path.join(MOMASK, 'checkpoints/t2m/Comp_v6_KLD005/meta')


# ─────────────────────────────────────────────── text -> GloVe tokens
def encode_caption(w_vectorizer, tokens, max_text_len=20):
    """tokens: list of 'word/POS' strings, as stored in HumanML3D texts/*.txt"""
    if len(tokens) < max_text_len:
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)

    pos_ohot, word_emb = [], []
    for t in tokens:
        we, po = w_vectorizer[t]
        word_emb.append(we[None, :])
        pos_ohot.append(po[None, :])
    return (np.concatenate(word_emb, 0), np.concatenate(pos_ohot, 0), sent_len)


def tag_with_spacy(sentences):
    """POS-tag raw sentences into 'word/POS' form, as HumanML3D does."""
    import spacy
    nlp = spacy.load('en_core_web_sm')
    out = []
    for s in sentences:
        doc = nlp(s.lower())
        toks = []
        for tok in doc:
            w = tok.text
            if not w.isalpha():
                continue
            toks.append(f'{w}/{tok.pos_}')
        out.append(toks)
    return out


# ─────────────────────────────────────────────── normalisation
def to_eval_space(motion_263, ds_mean, ds_std, ev_mean, ev_std,
                  already_denormed=False):
    """dataset-normalised (or raw) 263 -> evaluator-normalised 263"""
    m = motion_263 if already_denormed else motion_263 * ds_std + ds_mean
    return (m - ev_mean) / ev_std


def pad_to(motion, length=196):
    T = len(motion)
    if T >= length:
        return motion[:length], length
    pad = np.zeros((length - T, motion.shape[1]), dtype=motion.dtype)
    return np.concatenate([motion, pad], 0), T


# ─────────────────────────────────────────────── the evaluation
def embed_motions(evaluator, motions_eval_space, lens, batch=32, device='cuda:0'):
    """motions already in evaluator space and padded to 196"""
    embs = []
    for i in range(0, len(motions_eval_space), batch):
        M = torch.from_numpy(np.stack(motions_eval_space[i:i+batch])).float()
        L = torch.tensor(lens[i:i+batch])
        L = torch.clamp((L // 4) * 4, min=4)
        order = torch.argsort(L, descending=True)
        inv = torch.argsort(order)
        M, L = M[order], L[order]
        with torch.no_grad():
            mv = evaluator.movement_encoder(M.to(device)[..., :-4]).detach()
            e = evaluator.motion_encoder(mv, L // evaluator.opt.unit_length)
        embs.append(e.cpu().numpy()[inv.numpy()])
    return np.concatenate(embs)

def embed_texts(evaluator, w_vec, token_lists, batch=32, device='cuda:0'):
    embs = []
    for i in range(0, len(token_lists), batch):
        chunk = token_lists[i:i+batch]
        we, po, sl = zip(*[encode_caption(w_vec, t) for t in chunk])
        we = torch.from_numpy(np.stack(we)).float().to(device)
        po = torch.from_numpy(np.stack(po)).float().to(device)
        sl = torch.tensor(sl)
        order = torch.argsort(sl, descending=True)
        inv = torch.argsort(order)
        with torch.no_grad():
            e = evaluator.text_encoder(we[order], po[order], sl[order])
        embs.append(e.cpu().numpy()[inv.numpy()])
    return np.concatenate(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='mclip',
                    choices=['mclip', 'clip', 'both', 'original', 'beta010'])
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=500, help='test clips to evaluate')
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--replications', type=int, default=5)
    ap.add_argument('--langs', action='store_true',
                    help='also compute FID per language (no R-precision)')
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.device}")

    # ---- evaluator ----
    print('loading Guo evaluator...')
    ev = load_evaluator(dev)
    w_vec = load_word_vectorizer()
    ev_mean = np.load(os.path.join(EVAL_META, 'mean.npy'))
    ev_std = np.load(os.path.join(EVAL_META, 'std.npy'))
    ds_mean = np.load(os.path.join(args.datapath, 'Mean.npy'))
    ds_std = np.load(os.path.join(args.datapath, 'Std.npy'))
    ds_std[ds_std < 1e-6] = 1.0
    print(f'  evaluator stats {ev_mean.shape}, dataset stats {ds_mean.shape}')

    # ---- test set: motions + captions + tokens ----
    ids = [l.strip() for l in open(os.path.join(args.datapath, 'test.txt'))
           if l.strip()]
    ids = [c for c in ids if not c.startswith('M')]

    real, caps, toks, lens = [], [], [], []
    for cid in ids:
        mp = os.path.join(args.datapath, 'new_joint_vecs', cid + '.npy')
        tp = os.path.join(args.datapath, 'texts', cid + '.txt')
        if not (os.path.exists(mp) and os.path.exists(tp)):
            continue
        m = np.load(mp)
        if len(m) < 40:
            continue
        with cs.open(tp) as f:
            line = f.readline().strip().split('#')
        if len(line) < 2 or not line[0]:
            continue
        cap, tk = line[0], line[1].split(' ')
        tk = [t for t in tk if '/' in t]
        if not tk:
            continue
        mm, L = pad_to(to_eval_space(m, ds_mean, ds_std, ev_mean, ev_std,
                                     already_denormed=True))
        real.append(mm); lens.append(L); caps.append(cap); toks.append(tk)
        if len(real) >= args.n:
            break

    print(f'{len(real)} test clips loaded')

    from utils.metrics import (calculate_R_precision, calculate_frechet_distance,
                               calculate_activation_statistics,
                               calculate_matching_score, calculate_diversity)

    # ---- real baseline ----
    Zr_m = embed_motions(ev, real, lens, device=dev)
    Zr_t = embed_texts(ev, w_vec, toks, device=dev)
    mu_r, sig_r = calculate_activation_statistics(Zr_m)

    def rprec(Zt, Zm, reps, size=32, seed=0):
        rng = np.random.default_rng(seed)
        acc = []
        for _ in range(reps):
            idx = rng.permutation(len(Zm))[:(len(Zm)//size)*size]
            tot = np.zeros(3)
            n = 0
            for s in range(0, len(idx), size):
                g = idx[s:s+size]
                tot += calculate_R_precision(Zt[g], Zm[g], top_k=3, sum_all=True)
                n += size
            acc.append(tot / n)
        return np.mean(acc, 0), np.std(acc, 0)

    r_mean, r_std = rprec(Zr_t, Zr_m, args.replications)
    mm_real = float(np.mean(calculate_matching_score(Zr_t, Zr_m)))
    div_real = float(np.mean(calculate_diversity(Zr_m, min(300, len(Zr_m)-1))))
    print('\n' + '='*64)
    print(f'{"":22}{"R@1":>8}{"R@2":>8}{"R@3":>8}{"FID":>9}{"MM-Dist":>10}{"Div":>8}')
    print('='*64)
    print(f'{"real motion":22}{r_mean[0]:>8.3f}{r_mean[1]:>8.3f}{r_mean[2]:>8.3f}'
          f'{0.0:>9.3f}{mm_real:>10.3f}{div_real:>8.3f}')

    # ---- generated ----
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    CKPTS = {
        'original': (
            './exps/vae-en-mu-kl1e-3/checkpoint_0020.pth.tar',
            'mclip',
        ),
        'beta010': (
            './exps/vae-en-mu-beta10-ft/checkpoint_0025.pth.tar',
            'mclip',
        ),
    }
    tags = ['original', 'beta010'] if args.model == 'both' else (['original'] if args.model == 'original' else ['beta010'])

    for tag in tags:
        ckpt, enc_kind = CKPTS[tag]
        if not os.path.exists(ckpt):
            print(f'skip {tag}: {ckpt} missing')
            continue

        sys.argv = ['x', ckpt, '--input_file', 'assets/paper_texts.txt']
        params, folder, cname, epoch = vparser()
        params.update({'device': dev, 'dataset': 'humanml3d',
                       'datapath': args.datapath, 'pose_rep': 'xyz',
                       'text_encoder': enc_kind})
        model, _ = get_model_and_data(params, split='vald')
        load_model_wo_clip(model, torch.load(ckpt, map_location=dev))
        model.eval()

        gen, glens = [], []
        for s in range(0, len(caps), 16):
            chunk = caps[s:s+16]
            with torch.no_grad():
                f = model.clip_model.encode_text_raw(chunk).float().unsqueeze(0).to(dev)
                d = torch.ones((len(chunk), 1), dtype=int) * args.frames
                g = model.generate(f, d, is_amass=False, is_clip_features=True)
            out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)
            for o in out:
                mm, L = pad_to(to_eval_space(o, ds_mean, ds_std, ev_mean, ev_std))
                gen.append(mm); glens.append(L)
            del g
            torch.cuda.empty_cache()

        Zg = embed_motions(ev, gen, glens, device=dev)
        mu_g, sig_g = calculate_activation_statistics(Zg)
        fid = float(calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g))
        rg_mean, rg_std = rprec(Zr_t, Zg, args.replications)
        mmd = float(np.mean(calculate_matching_score(Zr_t, Zg)))
        div = float(np.mean(calculate_diversity(Zg, min(300, len(Zg)-1))))

        print(f'{tag:22}{rg_mean[0]:>8.3f}{rg_mean[1]:>8.3f}{rg_mean[2]:>8.3f}'
              f'{fid:>9.3f}{mmd:>10.3f}{div:>8.3f}')

        # ---- per-language FID, mclip only ----
        if args.langs and enc_kind == 'mclip':
            print('\n  per-language FID (R-precision is English-only --')
            print('  the evaluator\'s text tower is a GloVe BiGRU trained on English)')
            try:
                from deep_translator import GoogleTranslator
                LANGS = {'fr': 'french', 'es': 'spanish', 'de': 'german',
                         'ru': 'russian', 'ar': 'arabic', 'zh': 'chinese (simplified)',
                         'ja': 'japanese', 'hi': 'hindi'}
                sub = caps[:200]
                print(f'  {"lang":<10}{"FID":>9}{"vs en":>9}')
                for code, name in LANGS.items():
                    tr = GoogleTranslator(source='en', target=code).translate_batch(sub)
                    gl, ll = [], []
                    for s in range(0, len(tr), 16):
                        chunk = tr[s:s+16]
                        with torch.no_grad():
                            f = model.clip_model.encode_text_raw(chunk).float().unsqueeze(0).to(dev)
                            d = torch.ones((len(chunk), 1), dtype=int) * args.frames
                            g = model.generate(f, d, is_amass=False, is_clip_features=True)
                        out = g['output'].cpu().numpy()[:, :, 0, :].transpose(0, 2, 1)
                        for o in out:
                            mm, L = pad_to(to_eval_space(o, ds_mean, ds_std, ev_mean, ev_std))
                            gl.append(mm); ll.append(L)
                        del g; torch.cuda.empty_cache()
                    Zl = embed_motions(ev, gl, ll, device=dev)
                    mu_l, sig_l = calculate_activation_statistics(Zl)
                    f_l = float(calculate_frechet_distance(mu_r, sig_r, mu_l, sig_l))
                    print(f'  {code:<10}{f_l:>9.3f}{f_l - fid:>+9.3f}')
            except ImportError:
                print('  needs: pip install deep-translator')
            except Exception as e:
                print(f'  per-language failed: {type(e).__name__}: {e}')

        del model
        torch.cuda.empty_cache()

    print('\nLower FID and MM-Dist are better; higher R-precision is better.')
    print('Diversity should be close to the real-motion row, not maximised.')


if __name__ == '__main__':
    main()
