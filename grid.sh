#!/bin/bash

# Iterate over learning rates from 1e-3 to 1e-5
for lr in .0003 0.0001 .00003 0.00001; do
  # Iterate over lora_r values from 8 to 64
  for lora_r in 16 32 64; do
    # Iterate over scheduler types
    for scheduler in "linear" "cosine"; do
      # Set warmup steps based on the scheduler type and number of examples
      warmup_steps=200
      echo "Running with learning_rate=$lr, lora_r=$lora_r, and scheduler=$scheduler"
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python train.py --learning_rate $lr --lora_r $lora_r --scheduler_type $scheduler --warmup_steps $warmup_steps
    done
  done
done

