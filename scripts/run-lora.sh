export TOKENIZERS_PARALLELISM=false
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:25503 \
	train.py \
  --eval_every_steps 250 \
  --save_every_steps 250 \
  --alpha 100.0 \
  --lora \
  --lora_alpha 128 \
  --lora_r 128 \
  --lora_vision \
  --per_device_batch_size 2 \
  --eval_batch_size 2 \
  --learning_rate 5e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --max_eval_ids 250 \
  --use_flash_attn \
  --patch_prediction \
  --gradient_checkpointing \
  --dataset textvqa \
  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --train_on_questions \
#  --run_name "robust-fog-11"
#  --profile \
#  --use_packed_sampler \
