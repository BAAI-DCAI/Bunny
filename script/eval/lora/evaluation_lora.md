# Evaluation for LoRA tuning models

You can use `script/merge_lora_weights.py` to merge the LoRA weights and base LLM, and then evaluate it as [evaluation_full.md](../full/evaluation_full.md).

```Shell
python script/merge_lora_weights.py \
	--model-path /path/to/bunny_lora_weights \
	--model-base /path/to/base_llm_model \
	--model-type phi-2 (or stablelm-2 or phi-1.5 or qwen1.5-1.8b or minicpm or phi-3 or llama3-8b) \
	--save-model-path /path/to/merged_model
```

Or you can evaluate it without merging as below.

**Note that change `conv-mode` to `minicpm/phi3/llama` for `MODEL_TYPE = minicpm/phi-3/llama3-8b`.**

## MME

1. Refer to [MME GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) to download the benchmark dataset and put `MME_Benchmark_release_version` under `eval/mme`.
2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/mme.sh
```

The responses and scores can be found in `eval/mme/answers_upload`.

## MMBench & MMBench-Chinese

1. Refer to [MMBench GitHub](https://github.com/open-compass/MMBench) to download the benchmark dataset. We support `MMBench-Dev`, `MMBench-Test`, `MMBench-Dev (cn)` and `MMBench-Test (cn)`. Please note that only the files downloaded by **legacy link** are supported.
   Put `MMBench_DEV_EN_legacy.tsv`, `MMBench_TEST_EN_legacy.tsv`, `MMBench_DEV_CN_legacy.tsv` or `MMBench_TEST_CN_legacy.tsv` under `eval/mmbench`.
2. Update `SPLIT`, `LANG (en/cn)`, `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/mmbench.sh
```

The response file can be found in `eval/mmbench/answers_upload`. You can submit the Excel file to [submission link](https://mmbench.opencompass.org.cn/mmbench-submission) to obtain the evaluation scores.

## SEED-Bench-1

1. Refer to [SEED-Bench Instruction](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1) to download the images and videos and put the images under `eval/seed-bench/SEED-Bench-image` and the videos under `eval/seed-bench/SEED-Bench-video`. Then, extract the video frames in the middle from the downloaded videos by running:

   ```shell
   pip install av decord
   python eval/seed-bench/extract_video_frames.py
   ```


2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/lora/seedbench.sh
```

The response file can be found in `eval/seed-bench/answers_upload` and the scores can be found in `eval/seed-bench/scores`.

## MMMU

1. Refer to [MMMU HuggingFace](https://huggingface.co/datasets/MMMU/MMMU) to download the benchmark dataset and put `MMMU` under `eval/mmmu`.
2. Update `SPLIT`, `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly. You may add `--small-gpu-usage` to avoid `CUDA out of memory`.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/mmmu.sh
```

The response file can be found in `eval/mmmu/answers_upload`.

For validation set, you can use `eval_mmmu.py` to obtain the scores.

```shell
python eval/mmmu/eval_mmmu.py \
	--output-path ./eval/mmmu/answers_upload/$SPLIT/$TARGET_DIR.json
```

For test set, you can submit the `json` response file to [submission_link](https://eval.ai/web/challenges/challenge-page/2179/overview) to obtain the evaluation scores.

## CMMMU

1. Refer to [CMMMU HuggingFace](https://huggingface.co/datasets/m-a-p/CMMMU) to download the benchmark dataset and put `CMMMU` under `eval/cmmmu`.
2. Update `SPLIT`, `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly. You may add `--small-gpu-usage` to avoid `CUDA out of memory`.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/cmmmu.sh
```

The response file can be found in `eval/cmmmu/answers_upload`.

For validation set, you can use `eval_script.py` to obtain the scores.

```shell
python eval/cmmmu/eval_script.py \
	--output_path ./eval/cmmmu/answers_upload/$SPLIT/$TARGET_DIR.jsonl
```

For test set, you can submit the `jsonl` response file to [submission_link](https://eval.ai/web/challenges/challenge-page/2217/overview) to obtain the evaluation scores.

## VQAv2
1. Download [COCO 2015 Test images](http://images.cocodataset.org/zips/test2015.zip) and put `test2015` under `eval/vqav2`. Then:

   ```shell
   tar -zxvf eval/vqav2/bunny_vqav2_mscoco_test2015.tar.gz -C eval/vqav2 && rm eval/vqav2/bunny_vqav2_mscoco_test2015.tar.gz && tar -zxvf eval/vqav2/bunny_vqav2_mscoco_test-dev2015.tar.gz -C eval/vqav2 && rm eval/vqav2/bunny_vqav2_mscoco_test-dev2015.tar.gz
   ```

2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/lora/vqav2.sh
```

The response file can be found in `eval/vqav2/answers_upload`. You can submit the `json` response file to [submission link](https://eval.ai/web/challenges/challenge-page/830) (Test-Dev Phase) to obtain the evaluation scores.

## GQA
1. Download the [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) of GQA, unzip it and put `images` under `eval/gqa`. Then:

   ```shell
   tar -zxvf eval/gqa/testdev_balanced_questions.tar.gz -C eval/gqa && rm eval/gqa/testdev_balanced_questions.tar.gz
   ```

2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/lora/gqa.sh
```

## ScienceQA-IMG

1. Refer to [ScienceQA Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev) to download `test.zip`, `problems.json` and `pid_splits.json`, unzip `test.zip` and put them under `eval/scienceqa`.
2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/scienceqa.sh
```

The responses and the scores can be found in `eval/scienceqa/results`.

## POPE

1. Download [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip) and put `val2014` under `eval/pope`. Then, refer to [POPE GitHub](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) to download the benchmark dataset and put the three `json` files under `eval/pope/coco`.
2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/pope.sh
```

We report the averaged F1-score of three categories (random, popular and adversarial).

## MM-Vet

1. Refer to [MM-Vet Github](https://github.com/yuweihao/MM-Vet?tab=readme-ov-file#evalute-your-model-on-mm-vet) to download the benchmark dataset and put `images` under `eval/mm-vet`.
2. Update `MODEL_TYPE`, `MODEL_BASE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/lora/mmvet.sh
```

The response file can be found in `eval/mm-vet/answers_upload`. You can submit the `json` response file to [submission link](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) to obtain the evaluation scores.

