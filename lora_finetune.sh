#!/bin/bash

MODEL_ID="openai/whisper-base"
LEARNING_RATE="1e-4"

CUDA_VISIBLE_DEVICES=0 python lora_finetune.py \
    --train_data /home/jovyan/workspace/aura/corpus/audio/SAP/train.v2.jsonl \
    --test_data /home/jovyan/workspace/aura/corpus/audio/SAP/dev.v2.jsonl \
    --base_model $MODEL_ID \
    --learning_rate $LEARNING_RATE \
    --max_steps 10000 \
    --warmup_steps 500 \
    --lora_type lora \
    --language English \
    --task transcribe \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --output_dir output/lora \
    --push_to_hub False