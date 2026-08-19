"""
KL annealing: ramp lambda_kl from 0 to its target over the first N epochs.

    python patch_anneal.py            # apply
    python patch_anneal.py --revert

Why. Every KL weight tried so far leaves rc stuck at ~0.524 against 0.083 for
the deterministic run:

    kl 1e-5, with cosine     rc 0.524
    kl 1e-3, with cosine     rc 0.526
    kl 1e-5, no cosine       rc 0.524

A 100x change in the KL weight moved rc by 0.002, and removing the cosine loss
moved it by 0.000. So the damage is the sampling itself, not the weight.

From epoch 1 the decoder is asked to do two things at once: learn to
reconstruct, and learn to tolerate noise on the one vector carrying all 60
frames. Annealing separates them. lambda_kl starts at 0, so the first epochs
are a plain autoencoder and the decoder gets a clean signal; the noise is then
introduced gradually once reconstruction is established.

This is standard for VAEs that will not converge (Bowman et al. 2016 used it
for exactly this failure). It is the last cheap thing to try before concluding
the architecture cannot support a VAE.

    python -m src.train.train --lambda_kl 1e-3 --kl_anneal_epochs 30 ...

    epoch 1  -> lambda_kl = 0
    epoch 15 -> lambda_kl = 5e-4
    epoch 30 -> lambda_kl = 1e-3, and flat thereafter

If rc still sits at 0.52 with annealing, sampling is genuinely incompatible
with this bottleneck and that is the finding.
"""
import os
import sys
import shutil
import argparse

FILES = ['src/parser/model.py', 'src/train/train.py',
         'src/models/modeltype/motionclip.py']


def backup(p):
    b = p + '.pre_anneal'
    if not os.path.exists(b):
        shutil.copy(p, b)


def revert():
    n = 0
    for f in FILES:
        b = f + '.pre_anneal'
        if os.path.exists(b):
            shutil.copy(b, f); os.remove(b); n += 1
            print(f'reverted {f}')
    print(f'\n{n} reverted')


def apply():
    ok = True

    # ───────────── 1. flag
    p = 'src/parser/model.py'
    backup(p)
    s = open(p).read()
    if '--kl_anneal_epochs' in s:
        print(f'  --    {p}: flag present')
    else:
        anchor = '    group.add_argument("--lambda_kl"'
        if anchor in s:
            ins = ('    group.add_argument("--kl_anneal_epochs", default=0, type=int,\n'
                   '                       help="ramp lambda_kl linearly from 0 to its '
                   'target over this many epochs. 0 = no annealing")\n')
            s = s.replace(anchor, ins + anchor, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: --kl_anneal_epochs')
        else:
            print(f'  MISS  {p}: lambda_kl anchor not found')
            ok = False

    # ───────────── 2. model holds the target and current weight
    p = 'src/models/modeltype/motionclip.py'
    backup(p)
    s = open(p).read()
    if 'set_kl_weight' in s:
        print(f'  --    {p}: already has set_kl_weight')
    else:
        old = "        self.lambda_kl = float(lambdas.get('kl', 0.))"
        new = ("        self.lambda_kl = float(lambdas.get('kl', 0.))\n"
               "        self.lambda_kl_target = self.lambda_kl")
        if old in s:
            s = s.replace(old, new, 1)
        else:
            old2 = "        self.lambdas = lambdas"
            new2 = ("        self.lambdas = lambdas\n"
                    "        self.lambda_kl = float(lambdas.get('kl', 0.))\n"
                    "        self.lambda_kl_target = self.lambda_kl")
            if old2 in s:
                s = s.replace(old2, new2, 1)
            else:
                print(f'  MISS  {p}: cannot find where lambdas is stored')
                ok = False

        anchor = '    def reparameterize(self, batch, seed=None):'
        setter = '''    def set_kl_weight(self, epoch, anneal_epochs):
        """Linear ramp from 0 to the target over anneal_epochs.

        Note reparameterize() skips sampling while lambda_kl == 0, so epoch 1
        is a plain deterministic autoencoder and sampling switches on as soon
        as the weight leaves zero."""
        if anneal_epochs <= 0:
            self.lambda_kl = self.lambda_kl_target
        else:
            frac = min(1.0, epoch / float(anneal_epochs))
            self.lambda_kl = self.lambda_kl_target * frac
        if 'kl' in self.lambdas:
            self.lambdas['kl'] = self.lambda_kl
        return self.lambda_kl

    def reparameterize(self, batch, seed=None):'''
        if anchor in s:
            s = s.replace(anchor, setter, 1)
            print(f'  OK    {p}: set_kl_weight added')
        else:
            print(f'  MISS  {p}: reparameterize not found, apply patch_vae first')
            ok = False
        open(p, 'w').write(s)

    # ───────────── 3. call it each epoch
    p = 'src/train/train.py'
    backup(p)
    s = open(p).read()
    if 'set_kl_weight' in s:
        print(f'  --    {p}: already calls set_kl_weight')
    else:
        old = '        for epoch in range(start_epoch + 1, parameters["num_epochs"]+1):'
        new = ('''        anneal = parameters.get("kl_anneal_epochs", 0)
        for epoch in range(start_epoch + 1, parameters["num_epochs"]+1):
            if hasattr(model, "set_kl_weight"):
                w = model.set_kl_weight(epoch, anneal)
                if anneal > 0 and (epoch <= anneal or epoch % 10 == 0):
                    print(f"  lambda_kl = {w:.2e}")''')
        if old in s:
            s = s.replace(old, new, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: annealing applied per epoch')
        else:
            print(f'  MISS  {p}: epoch loop not as expected')
            ok = False

    print('\napplied' if ok else '\nAPPLIED WITH PROBLEMS, check MISS lines')
    if ok:
        print('\nkl_anneal_epochs = 0 behaves exactly as before, so nothing')
        print('already trained is affected.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if not os.path.exists('src/models/modeltype/motionclip.py'):
        sys.exit('run from ~/MotionCLIP')
    revert() if a.revert else apply()
