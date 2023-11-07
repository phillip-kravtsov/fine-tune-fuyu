export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2,3

torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:25502 \
	train.py \
  --eval_every_steps 100 \
  --save_every_steps 100 \
  --per_device_batch_size 12 \
  --eval_batch_size 12 \
  --learning_rate 3e-5 \
  --seed 102 \
  --use_flash_attn \
  --weight_decay 0.0 \
  --model_name_or_path "fuyu-8b-slim-vocab" \
  2>&1 | tee outputfile.txt
#  --max_eval_ids 200 \
#  --profile \
#  --use_packed_sampler \
