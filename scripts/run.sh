export TOKENIZERS_PARALLELISM=false

torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train.py \
  --eval_every_steps 1000 \
  --save_every_steps 100 \
  --per_device_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --use_flash_attn \
  --dataset scienceqa \
  --fsdp \
  --gradient_checkpointing \
#  --profile \
#  --use_packed_sampler \
#  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --max_eval_ids 200 \
