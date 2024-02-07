#!/bin/bash

MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

python -m bunny.eval.model_vqa_loader \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --image-folder ./eval/mme/MME_Benchmark_release_version \
    --question-file ./eval/mme/bunny_mme.jsonl \
    --answers-file ./eval/mme/answers/$TARGET_DIR.jsonl \
    --temperature 0 \
    --conv-mode bunny

cd ./eval/mme

python convert_answer_to_mme.py --experiment $TARGET_DIR

python calculation_mme.py --results_dir answers_upload/$TARGET_DIR \
| tee 2>&1 answers_upload/$TARGET_DIR/res.txt
