export TOKENIZERS_PARALLELISM=false

torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train.py \
  --eval_every_steps 10 \
  --save_every_steps 100 \
  --per_device_batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 2e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --use_flash_attn \
  --dataset ai2d \
  --lora \
  --gradient_checkpointing \
  --max_eval_ids 20 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
  --patch_prediction \
#  --ddp \
#  --fsdp \
#  --profile \
#  --use_packed_sampler \
