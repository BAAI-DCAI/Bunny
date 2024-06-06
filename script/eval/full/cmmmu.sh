#!/bin/bash

SPLIT="val"
MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

python -m bunny.eval.model_vqa_cmmmu \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --data-path ./eval/cmmmu/CMMMU \
    --config-path ./eval/cmmmu/prompt.yaml \
    --output-path ./eval/cmmmu/answers_upload/$SPLIT/$TARGET_DIR.jsonl \
    --split $SPLIT \
    --conv-mode bunny
