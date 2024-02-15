#!/bin/bash

MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

python -m bunny.eval.model_vqa_science \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --question-file ./eval/scienceqa/bunny_test_CQM-A.json \
    --image-folder ./eval/scienceqa/test \
    --answers-file ./eval/scienceqa/answers/$TARGET_DIR.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny

mkdir -p ./eval/scienceqa/outputs/
mkdir -p ./eval/scienceqa/results/

python ./eval/scienceqa/eval_science_qa.py \
    --base-dir ./eval/scienceqa \
    --result-file ./eval/scienceqa/answers/$TARGET_DIR.jsonl \
    --output-file ./eval/scienceqa/outputs/$TARGET_DIR.jsonl \
    --output-result ./eval/scienceqa/results/$TARGET_DIR.json