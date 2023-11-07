export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:25503 \
	train.py \
  --eval_every_steps 250 \
  --save_every_steps 250 \
  --lora \
  --lora_alpha 64 \
  --lora_r 64 \
  --lora_vision \
  --per_device_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 8e-5 \
  --seed 102 \
  --use_flash_attn \
  --weight_decay 0.0 \
  --use_packed_sampler \
  --model_name_or_path "fuyu-8b-slim-vocab" \
  2>&1 | tee outputfile.txt
#  --max_eval_ids 200 \
#  --profile \
