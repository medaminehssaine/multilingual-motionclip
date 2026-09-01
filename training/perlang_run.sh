#!/usr/bin/env bash
set -u
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate
echo "[$(date +%H:%M)] START perlang150-all"
python -m src.train.train \
  --folder ./exps/perlang150-all \
  --batch_size 128 --num_epochs 150 --lr 1e-4 --snapshot 50 \
  --dataset humanml3d_ml --datapath ./data/HumanML3D \
  --num_frames 60 --pose_rep xyz --cuda --device 0 \
  --text_encoder perlang --langs en fr es pt ru zh ar \
  --modelname motionclip_transformer_rc_rcxyz_vel \
  --latent_dim 512 --num_layers 8 --activation gelu \
  --jointstype vertices --glob --translation \
  --sampling conseq --sampling_step 1 \
  --lambda_rc 100 --lambda_rcxyz 0 --lambda_vel 100 --lambda_velxyz 1 \
  --clip_text_losses ce --clip_image_losses '' --clip_lambda_ce 10 \
  --lambda_kl 1e-3 --kl_anneal_epochs 0 \
  2>&1 | tee train_perlang150_all.log
echo "[$(date +%H:%M)] DONE perlang150-all"

# --- perlang + combined loss: completes the 2x2 (encoder x loss) ---
echo "[$(date +%H:%M)] START perlang150-combined"
python -m src.train.train \
  --folder ./exps/perlang150-combined \
  --batch_size 128 --num_epochs 150 --lr 1e-4 --snapshot 50 \
  --dataset humanml3d_ml --datapath ./data/HumanML3D \
  --num_frames 60 --pose_rep xyz --cuda --device 0 \
  --text_encoder perlang --langs en fr es pt ru zh ar \
  --modelname motionclip_transformer_rc_rcxyz_vel \
  --latent_dim 512 --num_layers 8 --activation gelu \
  --jointstype vertices --glob --translation \
  --sampling conseq --sampling_step 1 \
  --lambda_rc 100 --lambda_rcxyz 0 --lambda_vel 100 --lambda_velxyz 1 \
  --clip_text_losses ce_cosine --clip_image_losses '' \
  --clip_lambda_ce 10 --clip_lambda_cosine 10 \
  --lambda_kl 1e-3 --kl_anneal_epochs 0 \
  2>&1 | tee train_perlang150_combined.log

# --- KL sweep: 1e-3 was only ever compared against 0 ---
for KL in 1e-4 3e-3; do
  echo "[$(date +%H:%M)] START ce150-kl${KL}"
  python -m src.train.train \
    --folder "./exps/ce150-kl${KL}" \
    --batch_size 128 --num_epochs 150 --lr 1e-4 --snapshot 50 \
    --dataset humanml3d_ml --datapath ./data/HumanML3D \
    --num_frames 60 --pose_rep xyz --cuda --device 0 \
    --text_encoder mclip --langs en fr es pt ru zh ar \
    --modelname motionclip_transformer_rc_rcxyz_vel \
    --latent_dim 512 --num_layers 8 --activation gelu \
    --jointstype vertices --glob --translation \
    --sampling conseq --sampling_step 1 \
    --lambda_rc 100 --lambda_rcxyz 0 --lambda_vel 100 --lambda_velxyz 1 \
    --clip_text_losses ce --clip_image_losses '' --clip_lambda_ce 10 \
    --lambda_kl ${KL} --kl_anneal_epochs 0 \
    2>&1 | tee "train_ce150_kl${KL}.log"
done
echo "[$(date +%H:%M)] ALL DONE"
