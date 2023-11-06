torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train.py \
  --eval_every_steps 1 \
  --save_every_steps 100 \
  --per_device_batch_size 2 \
  --learning_rate 2e-5 \
  --seed 102 \
  --use_flash_attn \
  --weight_decay 0.0 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --use_packed_sampler \
#  --max_eval_ids 200 \
