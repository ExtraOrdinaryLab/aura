#!/bin/bash

MODEL_PATH="/path/to/checkpoint"
TEST_DATA="/path/to/SAP/dev.v2.jsonl"

python evaluation.py \
    --test_data $TEST_DATA \
    --model_path $MODEL_PATH
