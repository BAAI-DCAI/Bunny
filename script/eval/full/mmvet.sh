#!/bin/bash

MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

python -m bunny.eval.model_vqa \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --question-file ./eval/mm-vet/bunny-mm-vet.jsonl \
    --image-folder ./eval/mm-vet/images \
    --answers-file ./eval/mm-vet/answers/$TARGET_DIR.jsonl \
    --temperature 0 \
    --conv-mode bunny

mkdir -p ./eval/mm-vet/answers_upload

python ./eval/mm-vet/convert_mmvet_for_eval.py \
    --src ./eval/mm-vet/answers/$TARGET_DIR.jsonl \
    --dst ./eval/mm-vet/answers_upload/$TARGET_DIR.json