#!/bin/bash
cd ~/MotionCLIP
source ~/motionclip-env/bin/activate
python -m src.train.train --clip_text_losses cosine --clip_image_losses cosine --pose_rep rot6d \
--lambda_vel 100 --lambda_rc 100 --lambda_rcxyz 100 \
--jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 \
--lr 0.0001 --glob --translation --no-vertstrans --latent_dim 512 --num_epochs 100 --snapshot 10 \
--device 0 \
--dataset amass \
--datapath ./data/amass_db/amass_30fps_db.pt \
--folder ./exps/my-paper-model
