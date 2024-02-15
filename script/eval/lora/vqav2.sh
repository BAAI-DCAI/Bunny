#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

SPLIT="bunny_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m bunny.eval.model_vqa_loader \
        --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
        --model-base $MODEL_BASE \
        --model-type $MODEL_TYPE \
        --question-file ./eval/vqav2/$SPLIT.jsonl \
        --image-folder ./eval/vqav2/test2015 \
        --answers-file ./eval/vqav2/answers/$SPLIT/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode bunny &
done

wait

output_file=./eval/vqav2/answers/$SPLIT/$TARGET_DIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval/vqav2/answers/$SPLIT/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/vqav2/convert_vqav2_for_submission.py --split $SPLIT --ckpt $TARGET_DIR

