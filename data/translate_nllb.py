"""
Translate HumanML3D captions into six languages with NLLB-200, locally.

No API key, no rate limits, reproducible by anyone who clones the repo.

    python translate_nllb.py --extract          # captions_en.json
    python translate_nllb.py --lang fr          # one language
    python translate_nllb.py --all              # all six, sequential

Output: captions_<lang>.json, mapping the SAME clip ids as captions_en.json.

Notes:
  - only unique captions are translated (16,383 of 24,844 clips), then
    mapped back, which cuts the work by a third
  - writes partial results every --save_every batches, so a crash or a
    cancelled srun resumes rather than restarting
  - NLLB is a translation model, not a chat model: it takes a source
    language token and emits the target directly
"""
import os
import json
import time
import argparse
import codecs as cs

import torch

# NLLB uses FLORES-200 codes, not ISO-639-1
LANGS = {
    'fr': 'fra_Latn',      # French
    'es': 'spa_Latn',      # Spanish
    'pt': 'por_Latn',      # Portuguese
    'ru': 'rus_Cyrl',      # Russian
    'zh': 'zho_Hans',      # Chinese, simplified
    'ar': 'arb_Arab',      # Modern Standard Arabic
}
SRC = 'eng_Latn'

MODELS = {
    '600m': 'facebook/nllb-200-distilled-600M',
    '1.3b': 'facebook/nllb-200-distilled-1.3B',
    '3.3b': 'facebook/nllb-200-3.3B',
}


def extract(datapath, out='captions_en.json'):
    ids = []
    for split in ('train.txt', 'val.txt', 'test.txt'):
        p = os.path.join(datapath, split)
        if os.path.exists(p):
            ids += [l.strip() for l in open(p) if l.strip()]

    caps = {}
    for cid in ids:
        tp = os.path.join(datapath, 'texts', cid + '.txt')
        if not os.path.exists(tp):
            continue
        with cs.open(tp) as f:
            line = f.readline().strip().split('#')
        if line and line[0]:
            caps[cid] = line[0]

    json.dump(caps, open(out, 'w'), ensure_ascii=False, indent=0)
    uniq = set(caps.values())
    print(f'{len(caps)} clips, {len(uniq)} unique captions -> {out}')
    return caps


def translate_one(code, model_key, batch, device, save_every, maxlen):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tgt = LANGS[code]
    caps = json.load(open('captions_en.json'))
    uniq = sorted(set(caps.values()))

    out_path = f'captions_{code}.json'
    cache_path = f'.cache_{code}.json'
    done = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [c for c in uniq if c not in done]

    print(f'\n[{code}] {tgt}   {len(uniq)} unique, {len(done)} cached, '
          f'{len(todo)} to do')
    if not todo:
        print(f'[{code}] nothing to translate')
    else:
        name = MODELS[model_key]
        print(f'[{code}] loading {name}...')
        tok = AutoTokenizer.from_pretrained(name, src_lang=SRC)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()

        bos = tok.convert_tokens_to_ids(tgt)
        t0 = time.time()

        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc = tok(chunk, return_tensors='pt', padding=True,
                      truncation=True, max_length=maxlen).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos,
                                     max_length=maxlen, num_beams=1)
            out = tok.batch_decode(gen, skip_special_tokens=True)
            for src_txt, tgt_txt in zip(chunk, out):
                done[src_txt] = tgt_txt

            n = i + len(chunk)
            if (i // batch) % save_every == 0 or n >= len(todo):
                json.dump(done, open(cache_path, 'w'), ensure_ascii=False)
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                eta = (len(todo) - n) / rate / 60 if rate > 0 else 0
                print(f'  [{code}] {n}/{len(todo)}  {rate:.1f}/s  '
                      f'eta {eta:.0f}m')

        del model
        torch.cuda.empty_cache()

    # map unique translations back onto clip ids
    out_caps = {cid: done.get(en, en) for cid, en in caps.items()}
    missing = sum(1 for cid, en in caps.items() if en not in done)
    json.dump(out_caps, open(out_path, 'w'), ensure_ascii=False, indent=0)
    print(f'[{code}] wrote {out_path}  ({len(out_caps)} clips, '
          f'{missing} untranslated)')

    ex = list(caps.items())[:3]
    for cid, en in ex:
        print(f'    EN  {en[:70]}')
        print(f'    {code.upper()}  {out_caps[cid][:70]}')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--lang', nargs='*', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--datapath', default='./data/HumanML3D')
    ap.add_argument('--model', default='600m', choices=list(MODELS))
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--maxlen', type=int, default=128)
    ap.add_argument('--save_every', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    if args.extract or not os.path.exists('captions_en.json'):
        extract(args.datapath)
        if args.extract:
            return

    codes = list(LANGS) if args.all else (args.lang or [])
    if not codes:
        ap.error('give --lang fr [es ...] or --all')

    for c in codes:
        if c not in LANGS:
            print(f'unknown language {c}, skipping')
            continue
        translate_one(c, args.model, args.batch, args.device,
                      args.save_every, args.maxlen)

    print('\ndone')


if __name__ == '__main__':
    main()
