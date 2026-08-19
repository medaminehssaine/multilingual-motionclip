"""
Restore the VAE: sample the latent, add a KL term.

    python patch_vae.py            # apply
    python patch_vae.py --revert

MotionCLIP is ACTOR with the variational part switched off. The machinery is
all still present:

  transformer.py  computes mu AND logvar, then returns only mu
  motionclip.py   sets batch["z"] = batch["mu"], no sampling
  checkpoint.py   still references lambda_kl
  losses.py       has no kl entry, it was stripped

So this switches things back on rather than adding them:

  1. encoder returns logvar alongside mu
  2. reparameterise: z = mu + exp(0.5*logvar) * eps, training only
  3. compute_kl_loss registered in the loss table
  4. --lambda_kl flag, and kl appended to the loss list when non-zero

Why it matters here: the model is currently deterministic, so the 20-repeat
evaluation cannot measure sampling variance and the confidence intervals mean
something different from the published ones. A VAE samples, so they become
comparable. It is also the prerequisite for running diffusion over the latent.

Train:
    python -m src.train.train --lambda_kl 1e-5 ... --folder ./exps/vae-en

lambda_kl needs tuning. Too high and the latent collapses toward the prior and
reconstruction degrades; too low and sigma shrinks to zero and it is an
autoencoder again. Watch the kl term: if it goes to ~0 the model has collapsed
back to deterministic, if rc climbs sharply the KL is too strong.
"""
import os
import sys
import shutil
import argparse

FILES = [
    'src/models/architectures/transformer.py',
    'src/models/modeltype/motionclip.py',
    'src/models/tools/losses.py',
    'src/parser/model.py',
    'src/parser/training.py',
]


def backup(p):
    b = p + '.pre_vae'
    if not os.path.exists(b):
        shutil.copy(p, b)


def revert():
    n = 0
    for f in FILES:
        b = f + '.pre_vae'
        if os.path.exists(b):
            shutil.copy(b, f); os.remove(b); n += 1
            print(f'reverted {f}')
    print(f'\n{n} reverted')


def apply():
    ok = True

    # ───────────── 1. encoder returns logvar
    p = 'src/models/architectures/transformer.py'
    backup(p)
    s = open(p).read()
    if 'return {"mu": mu, "logvar": logvar}' in s:
        print(f'  --    {p}: already returns logvar')
    else:
        old = 'return {"mu": mu}'
        if old in s:
            s = s.replace(old, 'return {"mu": mu, "logvar": logvar}', 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: returns logvar')
        else:
            print(f'  MISS  {p}: return {{"mu": mu}} not found')
            ok = False

    # ───────────── 2. reparameterise in forward
    p = 'src/models/modeltype/motionclip.py'
    backup(p)
    s = open(p).read()
    if 'def reparameterize' in s:
        print(f'  --    {p}: already samples')
    else:
        old = '''        # encode
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]'''
        new = '''        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)'''
        if old in s:
            s = s.replace(old, new, 1)
        else:
            alt = '        batch["z"] = batch["mu"]'
            if alt in s:
                s = s.replace(alt, '        batch["z"] = self.reparameterize(batch)', 1)
            else:
                print(f'  MISS  {p}: batch["z"] = batch["mu"] not found')
                ok = False

        anchor = '    def forward(self, batch):'
        rep = '''    def reparameterize(self, batch, seed=None):
        """z = mu + sigma * eps, but only while training.

        At eval time the mean is used, which keeps generation reproducible and
        matches how the deterministic checkpoints behaved. Falls back to mu if
        logvar is absent, so old checkpoints still load."""
        mu = batch["mu"]
        logvar = batch.get("logvar", None)
        if logvar is None or not self.training or getattr(self, 'lambda_kl', 0.) == 0.:
            return mu
        std = torch.exp(0.5 * logvar)
        if seed is None:
            eps = torch.randn_like(std)
        else:
            g = torch.Generator(device=std.device).manual_seed(seed)
            eps = torch.randn(std.size(), device=std.device, generator=g)
        return mu + eps * std

    def forward(self, batch):'''
        if anchor in s:
            s = s.replace(anchor, rep, 1)
            print(f'  OK    {p}: reparameterize added')
        else:
            print(f'  MISS  {p}: forward not found')
            ok = False

        # remember lambda_kl on the model so reparameterize can check it
        old_l = "        self.lambdas = lambdas"
        new_l = ("        self.lambdas = lambdas\n"
                 "        self.lambda_kl = float(lambdas.get('kl', 0.))")
        if old_l in s and 'self.lambda_kl' not in s:
            s = s.replace(old_l, new_l, 1)
            print(f'  OK    {p}: lambda_kl stored on model')
        open(p, 'w').write(s)

    # ───────────── 3. kl loss function
    p = 'src/models/tools/losses.py'
    backup(p)
    s = open(p).read()
    if 'compute_kl_loss' in s:
        print(f'  --    {p}: kl already registered')
    else:
        fn = '''def compute_kl_loss(model, batch):
    """KL( q(z|x) || N(0,I) ), the standard closed form.

    Pushes each encoded distribution toward the unit gaussian, which is what
    stops the model shrinking sigma to zero and collapsing back into a plain
    autoencoder."""
    mu, logvar = batch["mu"], batch["logvar"]
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss / mu.shape[0]          # per-sample, so it does not scale with batch


'''
        anchor = 'def compute_hp_loss(model, batch):'
        if anchor in s:
            s = s.replace(anchor, fn + anchor, 1)
        else:
            print(f'  MISS  {p}: no anchor for the kl function')
            ok = False

        old_m = '"vel": compute_vel_loss, "velxyz": compute_velxyz_loss}'
        new_m = ('"vel": compute_vel_loss, "velxyz": compute_velxyz_loss,\n'
                 '              "kl": compute_kl_loss}')
        if old_m in s:
            s = s.replace(old_m, new_m, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: compute_kl_loss registered')
        else:
            print(f'  MISS  {p}: _matching_ table not as expected')
            ok = False

    # ───────────── 4. --lambda_kl
    p = 'src/parser/model.py'
    backup(p)
    s = open(p).read()
    if '--lambda_kl' in s:
        print(f'  --    {p}: --lambda_kl present')
    else:
        anchor = '    group.add_argument("--lambda_rc"'
        if anchor in s:
            ins = ('    group.add_argument("--lambda_kl", default=0.0, type=float,\n'
                   '                       help="weight of the KL term. 0 keeps the '
                   'model deterministic; try 1e-5 to 1e-3")\n')
            s = s.replace(anchor, ins + anchor, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: --lambda_kl')
        else:
            print(f'  MISS  {p}: lambda_rc anchor not found')
            ok = False

    # ───────────── 5. add kl to the loss list when non-zero
    p = 'src/parser/training.py'
    backup(p)
    s = open(p).read()
    if "'kl' not in parameters" in s or 'lambda_kl' in s:
        print(f'  --    {p}: kl already wired into the loss list')
    else:
        old = '''    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    parameters["lambdas"] = lambdas'''
        new = '''    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    # kl is not part of the model name, so append it when the weight is set
    if getattr(opt, "lambda_kl", 0.) > 0.:
        lambdas["kl"] = opt.lambda_kl
        if "kl" not in parameters["losses"]:
            parameters["losses"] = list(parameters["losses"]) + ["kl"]
        print(f"VAE mode: lambda_kl = {opt.lambda_kl}")
    parameters["lambdas"] = lambdas'''
        if old in s:
            s = s.replace(old, new, 1)
            open(p, 'w').write(s)
            print(f'  OK    {p}: kl appended when lambda_kl > 0')
        else:
            print(f'  MISS  {p}: lambdas block not as expected')
            ok = False

    print('\napplied' if ok else '\nAPPLIED WITH PROBLEMS, check MISS lines')
    if ok:
        print('\nlambda_kl = 0 reproduces the old deterministic behaviour exactly,')
        print('so existing checkpoints and runs are unaffected.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if not os.path.exists('src/models/tools/losses.py'):
        sys.exit('run from ~/MotionCLIP')
    revert() if a.revert else apply()
