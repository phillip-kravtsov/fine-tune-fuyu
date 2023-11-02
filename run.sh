torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 \
	train.py \
  --model_name_or_path "fuyu-8b-slim-vocab" \
  --eval_every_steps 500 \
  --per_device_batch_size 4
  #--model_name_or_path "../fuyu-2b" \
