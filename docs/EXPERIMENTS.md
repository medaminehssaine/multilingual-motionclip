# Experiment index

Every training run that produced a checkpoint, what it tested, and the headline
number. Section numbers refer to the internship report.

Only the final checkpoint of each run is kept, except where the epoch
progression is itself the result (`ce150-combined`, `ce150-all`,
`vae-ml-ce-s100-e100`, `vae-ml-ce-s100-e300`).

Two evaluators are used and their numbers are **not comparable**:
- **A** — Guo/MoMask on the official HumanML3D test split (`eval_momask.py`)
- **B** — `eval_guo_vae_compare_beta05.py`, 1400-clip validation pool

---

## 1. Multilingual baselines — report §4.1  (evaluator A, 20 replications)

| run | languages | FID ↓ | R@1 |
|---|---|---|---|
| `ml-en` | en | 18.013 | 0.128 |
| `ml-latin` | en fr es pt | **16.140** | **0.136** |
| `ml-far` | en ru zh ar | 18.046 | 0.111 |
| `ml-all` | all 7 | 17.829 | 0.119 |

Ground truth: FID 0.010, R@1 0.527, Diversity 9.465, MM-Dist 2.924.

Also here: `humanml3d-263`, `humanml3d-mclip`, `my-paper-model`, `paper-model`
— baseline reproductions of the original MotionCLIP before any modification.
`ml-en-s11` is an early English-only run (no CE loss, no `langs`); not cited.

## 2. Per-language encoder bank, first attempt — §4.2

| run | FID ↓ | R@1 | rc |
|---|---|---|---|
| `perlang-latin` | 24.55 | 0.079 | 0.086 |

Four frozen backbones (BERT / CamemBERT / BETO / BERTimbau), trainable 768→512
projections each. **Measured under the cosine objective, which §4.6 shows
produces no per-sample alignment in any condition — so this comparison is not
informative.** Re-run as `wk-perlang` (§9).

## 3. KL series, sampling active — §4.3

| run | change | rc ↓ |
|---|---|---|
| — | deterministic autoencoder | **0.083** |
| `vae-en-kl1e-5` | λ_KL = 1e-5 | 0.524 |
| `vae-en-kl1e-3` | λ_KL = 1e-3 | 0.526 |
| `vae-en-actor` | cosine term removed | 0.524 |
| `vae-en-anneal` | λ_KL annealed over 30 epochs | 0.525 |

Total spread 0.002 across four interventions. Sampling, not the KL weight, is
the binding constraint.

## 4. Deterministic reconstruction (from μ) — §4.5  (evaluator B)

| run | FID ↓ |
|---|---|
| `vae-en-mu-kl0` | 97.633 |
| `vae-en-mu-kl1e-3` | **55.864** |
| `vae-en-mu-kl1e-3-anneal30` | 105.683 |
| `vae-en-mu-kl1e-3-anneal5` | 100.893 |

rc recovers 0.52 → 0.16. KL helps; annealing hurts in both schedules.

## 5. Noise robustness — §4.6 (closed branch)

`vae-en-mu-beta02-ft`, `beta05-ft`, `beta10-ft`, `kl1e-3-beta05-ft`,
`kl1e-3-beta01`, `beta010-distill`, `beta010-decoderonly`, `beta010-musem`

β=0.10 fine-tuning: +25.5% reconstruction, 92% of clips improved, 95% CI
[0.227, 0.373]. Downstream FID degraded 57.317 → 86.774, R@1 0.062 → 0.031.
Distillation and encoder freezing did not recover it. Branch abandoned; the
negative transfer is the reported result.

## 6. Multilingual VAE and alignment — §4.7

| run | what it is | gap (en) |
|---|---|---|
| `vae-multilingual-mclip` | 7 languages, cosine loss | 0.0002 |
| `vae-ml-cos100` | cosine weight ×100 | 0.0007 |
| `vae-ml-ce` | contrastive CE, batch 64 | 0.0206 |
| `vae-ml-ce-s100` | CE, logit_scale 100, batch 128 | 0.0186 |

Matched vs mismatched cosine, 500 clips, 7 parallel captions. Raising the
cosine weight 100× made the collapse worse, not better: the objective has no
negatives, so a constant solution is a global optimum.

## 7. Contrastive alignment over training — §4.8

`vae-ml-ce-s100-e100` (5 checkpoints), `vae-ml-ce-s100-e300` (6 checkpoints)

Gap rises monotonically 0.0186 → 0.0286 (en, epochs 20→100), in every language,
no plateau at 300. Language ordering `es ≈ en > fr > pt > ar ≈ ru > zh` is
identical at every checkpoint.

Generation cost (evaluator B, CE-300): FID 72.902, Diversity 0.922 against
9.675 for real motion.

## 8. Variational fine-tune — §4.5

| run | sampling | rc ↓ |
|---|---|---|
| `vae-ml-ce-s100-e300` @300 | off | 0.093 |
| `vae-ml-sampled-ft` @330 | **on** | **0.086** |

Deterministic-to-convergence, then 10 epochs with sampling enabled.
**Caveat: two variables changed** — sampling was enabled *and* 10 further
epochs were run. Missing control: 10 more deterministic epochs from the same
checkpoint. Held-out confirmation across β also pending.

## 9. Amplitude and the text-conditioned path — §4.10, §4.11

Metric: mean absolute frame-to-frame difference of generated motion.
Second column: difference between prompts, as a conditioning measure.

| run | text_rc | λ_rc | encoder | amplitude | walk-vs-jump |
|---|---|---|---|---|---|
| `vae-en-mu-kl1e-3` (20 ep) | ✗ | 100 | M-CLIP | 0.0768 | 0.1127 |
| `vel10-test` (λ_vel 10) | ✗ | 100 | M-CLIP | 0.0145 | 0.0526 |
| `ce150-combined` | ✗ | 100 | M-CLIP | 0.0108 | 0.0409 |
| `wk-rc20` | ✗ | 20 | M-CLIP | 0.0078 | 0.0376 |
| `wk-textrc` | ✓ | 100 | M-CLIP | 0.0042 | **0.2456** |
| `wk-textrc-rc20` | ✓ | 20 | M-CLIP | 0.0030 | **0.2933** |
| `wk-perlang` | ✗ | 100 | per-lang | **0.0174** | 0.0934 |
| `wk-perlang-comb` | ✗ | 100 | per-lang | **0.0192** | 0.0972 |
| `vae-ml-sampled-ft` (330 ep) | ✗ | 100 | M-CLIP | 0.0044 | 0.0204 |

Amplitude decays monotonically with training length within a single run:

| epochs | `ce150-combined` | `ce150-all` |
|---|---|---|
| 50 | 0.01604 | 0.01626 |
| 100 | 0.01249 | 0.00929 |
| 150 | 0.01075 | 0.00626 |

Ruled out: rendering artefact, text-embedding scale mismatch (rescaling made it
worse, 0.01075 → 0.00607), velocity weight (matched-epoch comparison).

`text_rc` raises text conditioning 6–7× but reduces amplitude further — the
term rewards pose accuracy, not trajectory, since one embedding must explain
all 60 frames. Per-language encoders give 2–3× more amplitude than M-CLIP under
matched settings, reversing the apparent verdict of §4.2.

## 10. Controlled language comparison — §4.12

`ce150-en`, `ce150-latin`, `ce150-far`, `ce150-all`

150 epochs, batch 128, contrastive CE, identical except the language set.
Produced by `bash eval_all.sh`; raw output in `results/eval_all_ce150.log`.

| run | languages | gap (en − non-en) ↓ | spread ↓ | R@1 en / ar | FID ↓ | Diversity |
|---|---|---|---|---|---|---|
| `ce150-en` | en | +0.0318 | 0.0398 | 0.610 / 0.084 | 67.660 | 1.798 |
| `ce150-latin` | en fr es pt | +0.0087 | 0.0253 | 0.410 / 0.086 | 95.969 | 0.732 |
| `ce150-far` | en ru zh ar | +0.0059 | 0.0096 | 0.316 / 0.260 | **63.924** | 0.782 |
| `ce150-all` | all 7 | **+0.0019** | **0.0071** | 0.316 / 0.184 | 74.035 | 0.831 |

Real motion: R@1 0.529, MM-Dist 2.922, Diversity ≈ 9.5 (evaluator B).

Each language added to training closes its own gap, but English retrieval drops
from 0.610 to 0.316: the languages share a fixed latent budget rather than
enriching it. The `ce150-combined` (CE + cosine) block of the log is empty;
its alignment numbers are reported separately in the report.
