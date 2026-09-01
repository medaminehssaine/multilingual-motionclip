#!/usr/bin/env bash
# Evaluate every finished ce150-* condition: alignment + generation.
# Usage: bash eval_all.sh
set -u
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate

CONDS="ce150-combined ce150-en ce150-latin ce150-far ce150-all"
EP=0150

for c in $CONDS; do
  CK="exps/${c}/checkpoint_${EP}.pth.tar"
  if [ ! -f "$CK" ]; then
    echo "### SKIP ${c} -- no checkpoint yet"
    continue
  fi
  echo ""
  echo "###############################################################"
  echo "### ${c}"
  echo "###############################################################"

  echo "--- alignment ---"
  python eval_latent_alignment.py --ckpt "$CK" --tag "$c" --n 500 \
    --text_encoder mclip 2>&1 | grep -E "^(en|fr|es|pt|ru|zh|ar) |^gap|^spread|^english|^mean non"

  echo "--- generation (Guo evaluator, 20 reps) ---"
  python eval_guo_vae_compare_beta05.py --model original --replications 20 \
    --device 0 --ckpt "$CK" 2>&1 | grep -E "real motion|^original"
done

echo ""
echo "done $(date)"
