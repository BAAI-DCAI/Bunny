#!/bin/bash

SPLIT="test"
MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

python -m bunny.eval.model_vqa_mmmu \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --data-path ./eval/mmmu/MMMU \
    --config-path ./eval/mmmu/config.yaml \
    --output-path ./eval/mmmu/answers_upload/$SPLIT/$TARGET_DIR.json \
    --split $SPLIT \
    --conv-mode bunny
