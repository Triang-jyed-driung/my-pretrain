python main.py \
  --model_name ../rwkv-lm/RWKV-v5/out/fla \
  --config_args '{"fuse_norm":false,"fuse_cross_entropy":true}' \
  --data_file ../rwkv-lm/RWKV-v5/data/minipile \
  --compile_model \
  --rwkv_lr \
  --ctx_len 512 \
  --bsz 16 \
  --strategy deepspeed_stage_1 \
  --lr 6e-4 \
  --lr_end 6e-5 \
  --weight_decay 0.001 \
  --warmup_steps 10 \
  --grad_clip_val 1.0 \
  --accelerator gpu \
  --devices 1 \
  --save_every 10000000000 \
  --steps_per_epoch 2520 \
  --precision bf16-true \
  --wandb fla_7_align

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py \
#   --model_name /home/zhangping/zrc/fla_models/rwkv7-191m-world \
#   --config_args '{"fuse_norm":false,"fuse_cross_entropy":true}' \
#   --data_file /home/zhangping/zrc/zerocot/cot_corpus/cot_subsample_v2_text_document \
#   --compile_model \
#   --ctx_len 4096 \
#   --bsz 16 \
#   --strategy ddp \
#   --lr 2e-4 \
#   --lr_end 1e-4 \
#   --weight_decay 0.001 \
#   --warmup_steps 10 \
#   --grad_clip_val 4.0 \
#   --accelerator gpu \
#   --devices 6 \
#   --save_every 10 \
#   --steps_per_epoch 10000000 \
#   --precision bf16-true \
#   --wandb v7speed
  
  # --no_decay_1d
  # --config_args '{"fuse_cross_entropy":true}' \
  # --model_args {"_attn_implementation":"flash_attention_2"} \
  # --data_file /home/zhangping/zrc/RWKV-LM-v6/RWKV-v5/data/rwkv_mypile_v2 \
  # --compile_model \
