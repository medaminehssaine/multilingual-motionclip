# Engineering notes

## Environment

```bash
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate     # venv, NOT conda
```

Hardware used: single RTX 5090 (32 GB). Training at batch 128 runs ~32 s/epoch.

The machine is shared. Check the GPU before launching:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Long runs must be detached with `setsid`, not just `nohup`:

```bash
setsid nohup bash script.sh > out.log 2>&1 < /dev/null &
disown
```

`nohup ... & disown` alone did **not** survive an SSH disconnect on this
machine — two overnight queues were lost that way before `setsid` was used.

## Reproducing the headline numbers

```bash
# alignment gap, per language, any checkpoint
python eval_latent_alignment.py --ckpt exps/<run>/checkpoint_XXXX.pth.tar \
    --tag <name> --n 500 --text_encoder mclip     # or perlang

# generation quality (evaluator B)
python eval_guo_vae_compare_beta05.py --model original --replications 20 \
    --device 0 --ckpt exps/<run>/checkpoint_XXXX.pth.tar

# the controlled language comparison
bash eval_all.sh 2>&1 | tee results_all.log

# convergence figures
python plot_training_curves.py exps/<run1> exps/<run2>

# generation to GIF
python gen_hml.py "a person walks forward" "a person jumps up" \
    --ckpt exps/<run>/checkpoint_XXXX.pth.tar --outdir gen_out \
    --frames 60 --text_encoder mclip
```

## Known bugs and traps

**`reparameterize()` couples λ_KL to sampling.** In
`src/models/modeltype/motionclip.py`, sampling activates as soon as
`lambda_kl != 0`. Every KL-annealing run therefore trained with full-magnitude
noise from epoch 1 while only the *penalty* was ramped. This is why annealing
appeared to fail, and why deterministic-then-stochastic training was never
tested until late.

**Evaluation scripts hardcode checkpoints.** `eval_guo_vae_compare_beta05.py`
and `gen_hml.py` originally ignored the checkpoint you passed and silently
loaded a fixed one. Both now accept `--ckpt`. `eval_noise_val100.py` still
does **not** — it loads its own hardcoded pair, so any `--ckpt` given to it is
ignored. Fix before using it.

**`gen_hml.py` assumed plain CLIP.** It called `clip.tokenize` +
`encode_text`, which is wrong for M-CLIP and the per-language bank (both use
`encode_text_raw` on raw strings). Patched, with a `--text_encoder` flag that
**must** match how the checkpoint was trained — otherwise motion latents are
compared against the wrong text space and the output is meaningless without
raising an error.

**`logit_scale` is missing from both custom text encoders.** The CE path in
`compute_clip_losses` reads `self.clip_model.logit_scale`, which OpenAI CLIP
has and M-CLIP does not. Added as a learnable parameter to
`src/utils/mclip_wrapper.py` and `src/utils/per_lang_encoder.py`. It is frozen
in practice, since `get_model_and_data` freezes all CLIP parameters when
`clip_training == ''`.

**`--clip_text_losses` is underscore-separated**, not space-separated:
`--clip_text_losses ce_cosine`.

**Empty-string flags break inside shell variables.** Putting
`--clip_image_losses ''` inside a variable and expanding it unquoted drops the
empty argument entirely, and the next flag gets consumed as its value. Pass it
directly in each command.

**Training losses are not evidence.** The CE loss stayed near its chance value
(`ln(batch_size)`) while alignment improved by 100×, and reconstruction loss
looked healthy while generated motion collapsed to a near-static pose. Always
verify against an evaluation metric, not the loss curve.

## Tools written during this internship

| file | what it does |
|---|---|
| `eval_latent_alignment.py` | matched vs mismatched cosine + retrieval, 7 languages, per checkpoint |
| `plot_training_curves.py` | parses `training.log`, produces convergence figures |
| `patch_vae.py` | adds μ/logvar, reparameterisation, KL term |
| `patch_anneal.py` | KL annealing schedule |
| `patch_mclip.py` | CLIP → M-CLIP substitution |
| `patch_ml.py` | multilingual dataset with per-clip language sampling |
| `patch_perlang.py` | per-language frozen encoder bank |
| `translate_nllb.py` | NLLB caption translation into 6 target languages |
| `eval_all.sh` | evaluates every `ce150-*` condition, alignment + generation |

The `MOTIONCLIP_TEXT_RC` environment variable enables the text-conditioned
reconstruction term added in `compute_loss`. Default 0 (off).

## Open threads

1. **Re-run `eval_all.sh` for `ce150-combined`.** The EN / Latin / FAR / ALL
   comparison is done (`results/eval_all_ce150.log`, EXPERIMENTS §10), but the
   combined-loss block of that log is empty.
2. **Add velocity supervision to the text-conditioned path.** `text_rc` is
   currently pure positional MSE, which is likely why it improves conditioning
   6–7× while reducing amplitude. One flag, one run.
3. **Control for the VAE fine-tune.** Ten further deterministic epochs from
   `vae-ml-ce-s100-e300/checkpoint_0300`, to separate "sampling works" from
   "more training helps".
4. **Sequence of latents.** The architectural fix for the bottleneck: replace
   the single `muQuery` token in `Encoder_TRANSFORMER` with N tokens so noise
   on one latent damages only its frame span. Invalidates every checkpoint
   here; 2–3 days of work.
5. **Sweep λ_KL.** It was only ever compared against zero.
6. **Human-authored non-English captions.** All current ones are machine
   translations, so translation artefacts are confounded with genuine
   typological variation.

