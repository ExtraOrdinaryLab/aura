#!/bin/bash

CHECKPOINT_DIR="output/lora/sap/whisper-tiny/checkpoint-best"
OUTPUT_DIR="models/lora/sap"

python merge_lora.py \
    --lora_model $CHECKPOINT_DIR \
    --output_dir $OUTPUT_DIR \
    --local_files_only True