#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

SPLIT="bunny_gqa_testdev_balanced"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m bunny.eval.model_vqa_loader \
        --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
        --model-type $MODEL_TYPE \
        --question-file ./eval/gqa/$SPLIT.jsonl \
        --image-folder ./eval/gqa/images \
        --answers-file ./eval/gqa/answers/$SPLIT/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode bunny &
done

wait

output_file=./eval/gqa/answers/$SPLIT/$TARGET_DIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval/gqa/answers/$SPLIT/$TARGET_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/gqa/convert_gqa_for_eval.py --src $output_file --dst ./eval/gqa/testdev_balanced_predictions.json

cd eval/gqa
python eval_gqa.py --tier testdev_balanced
