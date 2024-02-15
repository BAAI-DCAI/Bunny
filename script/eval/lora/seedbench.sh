#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m bunny.eval.model_vqa_loader \
        --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
        --model-base $MODEL_BASE \
        --model-type $MODEL_TYPE \
        --question-file ./eval/seed-bench/bunny-seed-bench.jsonl \
        --image-folder ./eval/seed-bench \
        --answers-file ./eval/seed-bench/answers/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode bunny &
done

wait

output_file=./eval/seed-bench/answers/$TARGET_DIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval/seed-bench/answers/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./eval/seed-bench/answers_upload
mkdir -p ./eval/seed-bench/scores

# Evaluate
python ./eval/seed-bench/convert_seed_for_submission.py \
    --annotation-file ./eval/seed-bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./eval/seed-bench/answers_upload/$TARGET_DIR.jsonl | tee 2>&1 ./eval/seed-bench/scores/$TARGET_DIR.txt