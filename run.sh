torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=4 \
	train.py \
  --eval_every_steps 200 \
  --save_every_steps 400 \
  --max_eval_ids 200 \
  --per_device_batch_size 2 \
  --use_packed_sampler \
  --learning_rate 2e-5 \
  --seed 102 \
  --use_flash_attn \
  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --do_vocab_surgery \
#  --model_name_or_path "adept/fuyu-8b" \
#  --model_name_or_path "../fuyu-tiny" \
#--model_name_or_path "../fuyu-2b" \
