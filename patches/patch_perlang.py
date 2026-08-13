"""
Wire the per-language encoder bank into MotionCLIP.

    python patch_perlang.py            # apply
    python patch_perlang.py --revert

Then:
    python -m src.train.train --text_encoder perlang \
        --dataset humanml3d_ml --langs en fr es pt \
        --clip_image_losses '' ...

Four things this has to handle:

  1. The dataset returns a caption but not its language, so the caption is
     tagged "[fr] ..." and the encoder strips it. Keeps collate untouched.
  2. train.py drops every key starting with clip_model. when saving, which
     would silently discard the trained projections. Filter widened.
  3. clip.model.convert_weights() casts to fp16 and would break the HF
     backbones, so it is skipped for this encoder.
  4. The projections must exist before the optimizer is built, so they are
     constructed from --langs at model creation.
"""
import os
import sys
import shutil
import argparse

FILES = [
    'src/utils/get_model_and_data.py',
    'src/datasets/humanml3d_ml.py',
    'src/parser/training.py',
    'src/train/train.py',
]


def backup(p):
    b = p + '.pre_perlang'
    if not os.path.exists(b):
        shutil.copy(p, b)


def revert():
    n = 0
    for f in FILES:
        b = f + '.pre_perlang'
        if os.path.exists(b):
            shutil.copy(b, f); os.remove(b); n += 1
            print(f'reverted {f}')
    print(f'\n{n} reverted')


def apply():
    ok = True

    # ───────────────── 1. build the encoder, and skip fp16 conversion
    p = 'src/utils/get_model_and_data.py'
    backup(p)
    s = open(p).read()

    if 'perlang' in s:
        print(f'  --    {p}: already wired')
    else:
        old = "    if parameters.get('text_encoder', 'clip') == 'mclip':"
        new = """    if parameters.get('text_encoder', 'clip') == 'perlang':
        from src.utils.per_lang_encoder import PerLanguageTextEncoder
        clip_model = PerLanguageTextEncoder(
            langs=parameters.get('langs', ['en']),
            out_dim=parameters.get('latent_dim', 512),
            device=parameters['device'],
            encoders_device=parameters.get('encoders_device') or parameters['device'])
        clip_preprocess = None
        parameters['tag_language'] = True
        print(f"text encoder: per-language bank, "
              f"{len(clip_model.langs)} languages, {clip_model.out_dim}-dim")
    elif parameters.get('text_encoder', 'clip') == 'mclip':"""
        if old in s:
            s = s.replace(old, new, 1)
            print(f'  OK    {p}: perlang branch')
        else:
            print(f'  MISS  {p}: mclip branch not found')
            ok = False

        # convert_weights would cast the HF backbones to fp16 and break them
        oldc = "    clip.model.convert_weights(clip_model)"
        newc = ("    if parameters.get('text_encoder', 'clip') not in ('mclip', 'perlang'):\n"
                "        clip.model.convert_weights(clip_model)")
        if oldc in s:
            s = s.replace(oldc, newc, 1)
            print(f'  OK    {p}: convert_weights guarded')
        else:
            print(f'  WARN  {p}: convert_weights line not found, check manually')

        open(p, 'w').write(s)

    # ───────────────── 2. dataset tags the language
    p = 'src/datasets/humanml3d_ml.py'
    backup(p)
    s = open(p).read()
    if 'tag_language' in s:
        print(f'  --    {p}: already tags language')
    else:
        old = """    def get_clip_text(self, ind, frame_ix):
        lg = random.choice(self.langs)
        cap = self._caps[ind][lg]
        return np.array([cap] * len(frame_ix), dtype=object)"""
        new = """    def get_clip_text(self, ind, frame_ix):
        lg = random.choice(self.langs)
        cap = self._caps[ind][lg]
        # tag the language so the encoder can pick the right backbone.
        # a string prefix because the base collate only forwards the caption;
        # the encoder strips it before tokenising.
        if getattr(self, 'tag_language', False):
            cap = f'[{lg}] {cap}'
        return np.array([cap] * len(frame_ix), dtype=object)"""
        if old in s:
            s = s.replace(old, new, 1)
            s = s.replace("        self.langs = list(langs)",
                          "        self.langs = list(langs)\n"
                          "        self.tag_language = kargs.pop('tag_language', False)", 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: language tagging')
        else:
            print(f'  MISS  {p}: get_clip_text not as expected')
            ok = False

    # ───────────────── 3. flags
    p = 'src/parser/training.py'
    backup(p)
    s = open(p).read()
    if 'perlang' in s:
        print(f'  --    {p}: choices already include perlang')
    else:
        old = 'choices=["clip", "mclip"]'
        if old in s:
            s = s.replace(old, 'choices=["clip", "mclip", "perlang"]', 1)
            print(f'  OK    {p}: --text_encoder perlang')
        else:
            print(f'  MISS  {p}: text_encoder choices not found')
            ok = False

        anchor = 'group.add_argument("--langs"'
        if anchor in s and '--encoders_device' not in s:
            ins = ('group.add_argument("--encoders_device", default=None,\n'
                   '                       help="put the frozen backbones on cpu '
                   'to save GPU memory")\n    ')
            s = s.replace(anchor, ins + anchor, 1)
            print(f'  OK    {p}: --encoders_device')
        open(p, 'w').write(s)

    # ───────────────── 4. keep projections in the checkpoint
    p = 'src/train/train.py'
    backup(p)
    s = open(p).read()
    keep = "clip_model.projections."
    if keep in s:
        print(f'  --    {p}: save filter already widened')
    else:
        old = "if not k.startswith('clip_model.')"
        new = ("if not (k.startswith('clip_model.') and not "
               "(k.startswith('clip_model.projections.') or "
               "k.startswith('clip_model.heads.')))")
        if old in s:
            s = s.replace(old, new)
            open(p, 'w').write(s)
            print(f'  OK    {p}: projections kept in checkpoint')
        else:
            print(f'  MISS  {p}: save filter not found, '
                  f'projections will NOT be saved')
            ok = False

    print('\napplied' if ok else '\nAPPLIED WITH PROBLEMS, check the MISS lines')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if not os.path.exists('src/utils/get_model_and_data.py'):
        sys.exit('run from ~/MotionCLIP')
    revert() if a.revert else apply()
