#!/bin/bash

MODEL_PATHS=(
    "models/lora/sap/whisper-tiny-finetune"
)

TEST_MANIFESTS=(
    "/home/jovyan/workspace/aura/manifest/test_clean.jsonl"
    "/home/jovyan/workspace/aura/manifest/test_other.jsonl"
    "/home/jovyan/workspace/aura/manifest/atypical_severe.jsonl"
    "/home/jovyan/workspace/aura/manifest/atypical_moderate.jsonl"
    "/home/jovyan/workspace/aura/manifest/atypical_mild.jsonl"
    "/home/jovyan/workspace/aura/manifest/dev.v2.jsonl"
)

# Loop through each model and test data
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for TEST_MANIFEST in "${TEST_MANIFESTS[@]}"; do
        echo "Evaluating $MODEL_PATH with $TEST_MANIFEST"
        python evaluation.py \
            --test_data_path "$TEST_MANIFEST" \
            --model_path "$MODEL_PATH"
    done
done
