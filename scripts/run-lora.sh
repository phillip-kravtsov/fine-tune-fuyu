export TOKENIZERS_PARALLELISM=false
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:25503 \
	train.py \
  --eval_every_steps 1000 \
  --save_every_steps 250 \
  --lora \
  --lora_alpha 32 \
  --lora_r 32 \
  --lora_vision \
  --per_device_batch_size 4 \
  --eval_batch_size 1 \
  --learning_rate 5e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --max_eval_ids 200 \
  --use_flash_attn \
  --train_on_questions \
  --patch_prediction \
  --gradient_checkpointing \
  --dataset scienceqa \
  --model_name_or_path "fuyu-8b-slim-vocab"
#  --profile \
#  --use_packed_sampler \
