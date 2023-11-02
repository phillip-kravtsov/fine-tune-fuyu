torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 \
	train.py \
  --eval_every_steps 500 \
  --save_every_steps 500 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
  --per_device_batch_size 2 \
  --learning_rate 1e-4 \
  --max_eval_ids 200 \
  #--model_name_or_path "../fuyu-2b" \
