#!/bin/bash

CHECKPOINT_DIR="/path/to/checkpoint-best"

python merge_lora.py \
    --lora_model $CHECKPOINT_DIR \
    --output_dir models/lora