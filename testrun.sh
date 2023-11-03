torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:25499 \
	train.py \
  --eval_every_steps 10 \
  --save_every_steps 10 \
  --per_device_batch_size 2 \
  --use_packed_sampler \
  --learning_rate 1e-5 \
  --max_eval_ids 200 \
  --seed 102 \
  --model_name_or_path "fuyu-tiny-random"
# --use_flash_attn \
