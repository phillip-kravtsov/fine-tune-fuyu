torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:25499 \
	train.py \
  --eval_every_steps 1 \
  --save_every_steps 250 \
  --per_device_batch_size 2 \
  --learning_rate 1e-5 \
  --max_eval_ids 8 \
  --seed 102 \
  --model_name_or_path "fuyu-tiny-random" \
  --dataset ai2d \
  --use_flash_attn \
  --patch_prediction \
  --gradient_checkpointing \
  --lora \
#  --fsdp \
# --use_packed_sampler \