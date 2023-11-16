export TOKENIZERS_PARALLELISM=false

torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train.py \
  --eval_every_steps 250 \
  --save_every_steps 250 \
  --per_device_batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 5e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --use_flash_attn \
  --dataset textvqa \
  --fsdp \
  --gradient_checkpointing \
  --patch_prediction \
  --alpha 50.0 \
  --max_eval_ids 250 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --profile \
#  --use_packed_sampler \
