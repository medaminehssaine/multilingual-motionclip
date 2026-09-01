#!/usr/bin/env bash
set -u
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate
EPOCHS=150

run_condition () {
  local name="$1"; shift
  local losses="$1"; shift
  local lam_ce="$1"; shift
  local lam_cos="$1"; shift
  local langs="$*"

  if [ -f "./exps/${name}/checkpoint_0${EPOCHS}.pth.tar" ]; then
    echo "[$(date +%H:%M)] SKIP ${name} -- already done"
    return 0
  fi
  echo "=================================================="
  echo "[$(date +%H:%M)] START ${name} | losses=${losses} | langs: ${langs}"
  echo "=================================================="
  python -m src.train.train \
    --folder "./exps/${name}" \
    --batch_size 128 --num_epochs ${EPOCHS} --lr 1e-4 --snapshot 50 \
    --dataset humanml3d_ml --datapath ./data/HumanML3D \
    --num_frames 60 --pose_rep xyz --cuda --device 0 \
    --text_encoder mclip --langs ${langs} \
    --modelname motionclip_transformer_rc_rcxyz_vel \
    --latent_dim 512 --num_layers 8 --activation gelu \
    --jointstype vertices --glob --translation \
    --sampling conseq --sampling_step 1 \
    --lambda_rc 100 --lambda_rcxyz 0 --lambda_vel 100 --lambda_velxyz 1 \
    --clip_text_losses "${losses}" --clip_image_losses '' \
    --clip_lambda_ce ${lam_ce} --clip_lambda_cosine ${lam_cos} \
    --lambda_kl 1e-3 --kl_anneal_epochs 0 \
    2>&1 | tee "train_${name}.log"
  echo "[$(date +%H:%M)] DONE ${name} (exit ${PIPESTATUS[0]})"
}

echo "queue started $(date)"
run_condition ce150-combined ce_cosine 10 10  en fr es pt ru zh ar
run_condition ce150-en       ce        10 0   en
run_condition ce150-latin    ce        10 0   en fr es pt
run_condition ce150-far      ce        10 0   en ru zh ar
run_condition ce150-all      ce        10 0   en fr es pt ru zh ar
echo "queue finished $(date)"
ls -lh exps/ce150-*/checkpoint_0${EPOCHS}.pth.tar 2>/dev/null

echo "queue finished $(date)"
