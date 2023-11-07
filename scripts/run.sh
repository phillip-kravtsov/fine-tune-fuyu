export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train.py \
  --eval_every_steps 100 \
  --save_every_steps 100 \
  --per_device_batch_size 6 \
  --eval_batch_size 4 \
  --learning_rate 2e-5 \
  --seed 102 \
  --weight_decay 0.0 \
  --max_eval_ids 200 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
#  --profile \
#  --use_packed_sampler \
#  --use_flash_attn \
