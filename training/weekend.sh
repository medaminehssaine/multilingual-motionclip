#!/usr/bin/env bash
set -u
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate
echo "=== started $(date) ==="

C="--batch_size 128 --lr 1e-4 --snapshot 50 --num_epochs 150 \
 --dataset humanml3d_ml --datapath ./data/HumanML3D \
 --num_frames 60 --pose_rep xyz --cuda --device 0 \
 --langs en fr es pt ru zh ar \
 --modelname motionclip_transformer_rc_rcxyz_vel \
 --latent_dim 512 --num_layers 8 --activation gelu \
 --jointstype vertices --glob --translation \
 --sampling conseq --sampling_step 1 \
 --lambda_rcxyz 0 --lambda_velxyz 1 \
 --lambda_kl 1e-3 --kl_anneal_epochs 0"

run () { echo "[$(date +%H:%M)] START $1"; shift; "$@" ; echo "[$(date +%H:%M)] DONE ($?)"; }

# 1 -- text-conditioned reconstruction: trains the generation path itself
echo "[$(date +%H:%M)] START wk-textrc"
MOTIONCLIP_TEXT_RC=100 python -m src.train.train --folder ./exps/wk-textrc \
  --text_encoder mclip $C --lambda_rc 100 --lambda_vel 100 \
  --clip_image_losses '' --clip_text_losses ce_cosine --clip_lambda_ce 10 --clip_lambda_cosine 10 \
  2>&1 | tee train_wk_textrc.log

# 2 -- weaker reconstruction: tests whether MSE dominance causes the flattening
echo "[$(date +%H:%M)] START wk-rc20"
python -m src.train.train --folder ./exps/wk-rc20 \
  --text_encoder mclip $C --lambda_rc 20 --lambda_vel 100 \
  --clip_image_losses '' --clip_text_losses ce_cosine --clip_lambda_ce 10 --clip_lambda_cosine 10 \
  2>&1 | tee train_wk_rc20.log

# 3 -- per-language encoders, CE  (Ismail's literal question)
echo "[$(date +%H:%M)] START wk-perlang"
python -m src.train.train --folder ./exps/wk-perlang \
  --text_encoder perlang $C --lambda_rc 100 --lambda_vel 100 \
  --clip_image_losses '' --clip_text_losses ce --clip_lambda_ce 10 \
  2>&1 | tee train_wk_perlang.log

# 4 -- per-language encoders, combined loss (completes the 2x2)
echo "[$(date +%H:%M)] START wk-perlang-comb"
python -m src.train.train --folder ./exps/wk-perlang-comb \
  --text_encoder perlang $C --lambda_rc 100 --lambda_vel 100 \
  --clip_image_losses '' --clip_text_losses ce_cosine --clip_lambda_ce 10 --clip_lambda_cosine 10 \
  2>&1 | tee train_wk_perlang_comb.log

# 5 -- text_rc with weaker motion rc: both levers at once
echo "[$(date +%H:%M)] START wk-textrc-rc20"
MOTIONCLIP_TEXT_RC=100 python -m src.train.train --folder ./exps/wk-textrc-rc20 \
  --text_encoder mclip $C --lambda_rc 20 --lambda_vel 100 \
  --clip_image_losses '' --clip_text_losses ce_cosine --clip_lambda_ce 10 --clip_lambda_cosine 10 \
  2>&1 | tee train_wk_textrc_rc20.log

echo "=== finished $(date) ==="
ls -lh exps/wk-*/checkpoint_0150.pth.tar 2>/dev/null
