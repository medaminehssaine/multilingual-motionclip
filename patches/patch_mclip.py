"""
Add M-CLIP as a swappable text encoder in MotionCLIP.

Run from ~/MotionCLIP:
    python patch_mclip.py            # apply
    python patch_mclip.py --revert   # undo

What it changes:

  1. new file  src/utils/mclip_wrapper.py
        wraps M-CLIP so it exposes encode_text_raw(list_of_strings),
        frozen, on the right device.

  2. src/models/modeltype/motionclip.py
        both text-encoding call sites go through encode_text_raw() instead
        of clip.tokenize() + encode_text(). For plain CLIP this is the same
        two lines it already ran; for M-CLIP it skips CLIP's tokenizer.

  3. src/utils/get_model_and_data.py
        loads M-CLIP instead of CLIP when parameters['text_encoder'] == 'mclip'.
        Image encoder is still CLIP's, but with M-CLIP the image loss must be
        off -- the two live in different spaces.

  4. src/parser/training.py
        adds --text_encoder {clip,mclip}

After patching, train with:

    python -m src.train.train \
      --text_encoder mclip \
      --clip_text_losses cosine --clip_image_losses '' \
      --pose_rep xyz --lambda_rc 100 --lambda_rcxyz 0 --lambda_vel 100 \
      --jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 \
      --lr 0.0001 --glob --translation --no-vertstrans --latent_dim 512 \
      --num_epochs 100 --snapshot 10 --device 0 \
      --dataset humanml3d --datapath ./data/HumanML3D \
      --folder ./exps/humanml3d-mclip
"""
import os
import sys
import shutil
import argparse

FILES = [
    'src/models/modeltype/motionclip.py',
    'src/utils/get_model_and_data.py',
    'src/parser/training.py',
]

WRAPPER = '''import torch
import torch.nn as nn


class MClipTextEncoder(nn.Module):
    """M-CLIP as a drop-in text encoder.

    MotionCLIP only ever needs text -> 512-dim vector, so that is all this
    exposes. Frozen: we are aligning motion to this space, not moving it.
    """

    def __init__(self, name='M-CLIP/XLM-Roberta-Large-Vit-B-32', device='cuda:0'):
        super().__init__()
        from multilingual_clip import pt_multilingual_clip
        import transformers

        self.name = name
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._device = device

        with torch.no_grad():
            self.out_dim = self.model.forward(['probe'], self.tokenizer).shape[-1]

    def encode_text_raw(self, texts):
        texts = [str(t) for t in texts]
        return self.model.forward(texts, self.tokenizer).to(self._device)

    def encode_text(self, tokens):
        raise RuntimeError('M-CLIP takes raw strings; use encode_text_raw()')

    def encode_image(self, images):
        raise RuntimeError('M-CLIP has no image encoder -- run with '
                           "--clip_image_losses ''")


def attach_encode_text_raw(clip_model):
    """Give a stock OpenAI CLIP model the same encode_text_raw() interface."""
    import clip as openai_clip

    def encode_text_raw(texts):
        dev = next(clip_model.parameters()).device
        tok = openai_clip.tokenize([str(t) for t in texts], truncate=True).to(dev)
        return clip_model.encode_text(tok)

    clip_model.encode_text_raw = encode_text_raw
    return clip_model
'''


def backup(path):
    b = path + '.pre_mclip'
    if not os.path.exists(b):
        shutil.copy(path, b)


def revert():
    n = 0
    for f in FILES:
        b = f + '.pre_mclip'
        if os.path.exists(b):
            shutil.copy(b, f)
            os.remove(b)
            print(f'reverted {f}')
            n += 1
    w = 'src/utils/mclip_wrapper.py'
    if os.path.exists(w):
        os.remove(w)
        print(f'removed {w}')
    print(f'\n{n} file(s) reverted')


def apply():
    ok = True

    # ---- 1. wrapper ----
    with open('src/utils/mclip_wrapper.py', 'w') as f:
        f.write(WRAPPER)
    print('wrote  src/utils/mclip_wrapper.py')

    # ---- 2. motionclip.py: route both call sites through encode_text_raw ----
    p = 'src/models/modeltype/motionclip.py'
    backup(p)
    s = open(p).read()

    old = ("texts = clip.tokenize(batch['clip_text'], truncate=True).to(self.device)\n")
    n_tok = s.count(old.strip())
    if n_tok == 0:
        print(f'  MISS  {p}: tokenize call not found')
        ok = False
    else:
        # replace "tokenize(...)  /  encode_text(texts)" pairs
        import re
        pat = re.compile(
            r"texts = clip\.tokenize\(batch\['clip_text'\], truncate=True\)\.to\(self\.device\)\s*\n"
            r"(\s*)(\w+) = self\.clip_model\.encode_text\(texts\)")
        s2, n = pat.subn(
            lambda m: f"{m.group(1)}{m.group(2)} = self.clip_model.encode_text_raw(batch['clip_text'])",
            s)
        if n == 0:
            # fall back: patch the two lines separately
            s2 = s.replace(
                "texts = clip.tokenize(batch['clip_text'], truncate=True).to(self.device)",
                "texts = batch['clip_text']")
            s2 = s2.replace("self.clip_model.encode_text(texts)",
                            "self.clip_model.encode_text_raw(texts)")
            print(f'  OK    {p}: patched via fallback')
        else:
            print(f'  OK    {p}: {n} call site(s) rerouted')
        open(p, 'w').write(s2)

    # ---- 3. get_model_and_data.py ----
    p = 'src/utils/get_model_and_data.py'
    backup(p)
    s = open(p).read()
    old = 'clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters[\'device\'], jit=False)'
    if old not in s:
        print(f'  MISS  {p}: clip.load line not found')
        ok = False
    else:
        new = '''if parameters.get('text_encoder', 'clip') == 'mclip':
        from src.utils.mclip_wrapper import MClipTextEncoder
        clip_model = MClipTextEncoder(device=parameters['device'])
        clip_preprocess = None
        print(f"text encoder: M-CLIP ({clip_model.name}), {clip_model.out_dim}-dim")
        if clip_model.out_dim != parameters.get('latent_dim', 512):
            print(f"  WARNING: latent_dim is {parameters.get('latent_dim')}, "
                  f"M-CLIP emits {clip_model.out_dim}")
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)
        from src.utils.mclip_wrapper import attach_encode_text_raw
        attach_encode_text_raw(clip_model)'''
        s = s.replace(old, new, 1)
        open(p, 'w').write(s)
        print(f'  OK    {p}: encoder switch added')

    # ---- 4. training parser ----
    p = 'src/parser/training.py'
    backup(p)
    s = open(p).read()
    anchor = "group.add_argument('--clip_text_losses'"
    if anchor not in s:
        print(f'  MISS  {p}: could not find an anchor for the new flag')
        ok = False
    else:
        ins = ("group.add_argument('--text_encoder', default='clip', "
               "choices=['clip', 'mclip'],\n"
               "                       help=\"text encoder; 'mclip' is "
               "multilingual and has no image tower\")\n    ")
        s = s.replace(anchor, ins + anchor, 1)
        open(p, 'w').write(s)
        print(f'  OK    {p}: --text_encoder added')

    print('\napplied' if ok else '\napplied with problems -- check the MISS lines above')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if not os.path.exists('src/models/modeltype/motionclip.py'):
        sys.exit('run this from ~/MotionCLIP')
    revert() if a.revert else apply()
