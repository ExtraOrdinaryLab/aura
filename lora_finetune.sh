#!/bin/bash

MODEL_ID="openai/whisper-tiny"
TRAIN_MANIFEST="/home/jovyan/workspace/aura/manifest/fisher_segments.jsonl"
TEST_MANIFEST="/home/jovyan/workspace/aura/manifest/eval2000.jsonl"

LEARNING_RATE="1e-4"

OUTPUT_DIR="output/lora/sap"

# Load environment variables from .env file
set -o allexport
source .env
set +o allexport

# Now you can use the variables
echo "HF_TOKEN is: $HF_TOKEN"
echo "WANDB_PROJECT is $WANDB_PROJECT"
echo "WANDB_API_KEY is: $WANDB_API_KEY"

CUDA_VISIBLE_DEVICES=0 python lora_finetune.py \
    --train_data $TRAIN_MANIFEST \
    --test_data $TEST_MANIFEST \
    --base_model $MODEL_ID \
    --learning_rate $LEARNING_RATE \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_steps 10000 \
    --warmup_steps 500 \
    --lora_type lora \
    --language English \
    --task transcribe \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --output_dir $OUTPUT_DIR \
    --push_to_hub False