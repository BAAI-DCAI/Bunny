#!/bin/bash

SPLIT="MMBench_DEV_EN_legacy"
LANG=en
MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2


python -m bunny.eval.model_vqa_mmbench \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --question-file ./eval/mmbench/$SPLIT.tsv \
    --answers-file ./eval/mmbench/answers/$SPLIT/$TARGET_DIR.jsonl \
    --lang $LANG \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny

mkdir -p eval/mmbench/answers_upload/$SPLIT

python eval/mmbench/convert_mmbench_for_submission.py \
    --annotation-file ./eval/mmbench/$SPLIT.tsv \
    --result-dir ./eval/mmbench/answers/$SPLIT \
    --upload-dir ./eval/mmbench/answers_upload/$SPLIT \
    --experiment $TARGET_DIR
