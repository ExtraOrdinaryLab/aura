#!/bin/bash

MODEL_ID="openai/whisper-medium"
TRAIN_MANIFEST="/home/jovyan/workspace/aura/manifest/sap_train.jsonl"
TEST_MANIFEST="/home/jovyan/workspace/aura/manifest/sap_dev.jsonl"

LEARNING_RATE="5e-6"

OUTPUT_DIR="output/full/sap"

# Load environment variables from .env file
set -o allexport
source .env
set +o allexport

# Now you can use the variables
echo "HF_TOKEN is: $HF_TOKEN"
echo "WANDB_PROJECT is $WANDB_PROJECT"
echo "WANDB_API_KEY is: $WANDB_API_KEY"
    
CUDA_VISIBLE_DEVICES=0 python full_finetune.py \
    --train_data $TRAIN_MANIFEST \
    --test_data $TEST_MANIFEST \
    --base_model $MODEL_ID \
    --learning_rate $LEARNING_RATE \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_steps 10000 \
    --warmup_steps 500 \
    --language English \
    --task transcribe \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --output_dir $OUTPUT_DIR \
    --save_total_limit 1 \
    --report_to "none" \
    --push_to_hub False
    