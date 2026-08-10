"""
Register HumanML3DML and add --langs to the training parser.

    python patch_ml.py            # apply
    python patch_ml.py --revert

Then:
    # controlled English baseline (first-caption only, same pool)
    python -m src.train.train --dataset humanml3d_ml --langs en ...

    # multilingual
    python -m src.train.train --dataset humanml3d_ml \
        --langs en fr es pt ru zh ar ...
"""
import os
import sys
import shutil
import argparse

FILES = ['src/datasets/get_dataset.py', 'src/parser/training.py']


def backup(p):
    b = p + '.pre_ml'
    if not os.path.exists(b):
        shutil.copy(p, b)


def revert():
    for f in FILES:
        b = f + '.pre_ml'
        if os.path.exists(b):
            shutil.copy(b, f)
            os.remove(b)
            print(f'reverted {f}')
    print('done')


def apply():
    ok = True

    # ---- 1. register the dataset ----
    p = 'src/datasets/get_dataset.py'
    backup(p)
    s = open(p).read()

    if 'HumanML3DML' in s:
        print(f'  --    {p}: already registered')
    else:
        if 'from .humanml3d import HumanML3D' not in s:
            print(f'  MISS  {p}: humanml3d import not found; '
                  f'apply the humanml3d patch first')
            ok = False
        else:
            s = s.replace('from .humanml3d import HumanML3D',
                          'from .humanml3d import HumanML3D\n'
                          'from .humanml3d_ml import HumanML3DML', 1)
            if 'DATASETS = {' in s:
                s = s.replace('"humanml3d": HumanML3D}',
                              '"humanml3d": HumanML3D,\n'
                              '            "humanml3d_ml": HumanML3DML}', 1)
            else:
                print(f'  MISS  {p}: DATASETS dict not found')
                ok = False
            open(p, 'w').write(s)
            print(f'  OK    {p}: HumanML3DML registered')

    # ---- 2. --langs flag ----
    p = 'src/parser/training.py'
    backup(p)
    s = open(p).read()

    if '--langs' in s:
        print(f'  --    {p}: --langs already present')
    else:
        anchor = 'group.add_argument("--text_encoder"'
        if anchor not in s:
            anchor = 'group.add_argument("--batch_size"'
        if anchor not in s:
            print(f'  MISS  {p}: no anchor found for --langs')
            ok = False
        else:
            ins = ('group.add_argument("--langs", nargs="+", default=["en"],\n'
                   '                       help="caption languages to sample '
                   'from, for dataset humanml3d_ml")\n'
                   '    group.add_argument("--caption_dir", default=".",\n'
                   '                       help="where captions_<lang>.json live")\n'
                   '    ')
            s = s.replace(anchor, ins + anchor, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: --langs and --caption_dir added')

    print('\napplied' if ok else '\napplied with problems')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if not os.path.exists('src/datasets/get_dataset.py'):
        sys.exit('run from ~/MotionCLIP')
    revert() if a.revert else apply()
