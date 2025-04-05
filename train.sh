CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name fla-hub/rwkv7-0.1B-g1 \
  --compile_model \
  --data_file /home/zhangping/zrc/RWKV-LM-v6/RWKV-v5/data/rwkv_mypile_v2 \
  --ctx_len 4096 \
  --bsz 28 \
  --strategy ddp \
  --lr 1e-4 \
  --lr_end 5e-6 \
  --weight_decay 0.1 \
  --warmup_steps 10 \
  --accelerator gpu \
  --devices 1 \
  --save_every 10 \
  --samples_per_epoch 1290240 \
  --precision bf16-true \
  --wandb r7_0b1_g1 \
  --no_decay_1d

  # --config_args '{"fuse_cross_entropy":true}' \
  # --model_args {"_attn_implementation":"flash_attention_2"} \
