# Bunny: A family of lightweight multimodal models

<p align="center">
  <img src="./icon.png" alt="Logo" width="350">
</p>

📖 Technical report (coming soon) | 🤗 [Models](#model-zoo) | 🐰 [Demo](http://bunny.dataoptim.org)

Bunny is a family of lightweight but powerful multimodal models. It offers multiple plug-and-play vision encoders, like EVA-CLIP, SigLIP and language backbones, including Phi-1.5, StableLM-2 and Phi-2. To compensate for the decrease in model size, we construct more informative training data by curated selection from a broader data source. Remarkably, our Bunny-3B model built upon SigLIP and Phi-2 outperforms the state-of-the-art MLLMs, not only in comparison with models of similar size but also against larger MLLMs (7B), and even achieves performance on par with 13B models.

![comparison](comparison.png)

## News and Updates

* ⏳ Bunny training data.
* ⏳ Bunny technical report.
* 2024.2.7 🔥  **Bunny is released!** Bunny-3B built upon SigLIP and Phi-2 outperforms the state-of-the-art MLLMs, not only in comparison with models of similar size but also against larger MLLMs (7B), and even achieves performance on par with LLaVA-13B!

## Model Zoo

* Evaluation
  
| Checkpoint                                                   | MME$`^\text{P}`$ | MME$`^\text{C}`$ | MMB$`^\text{T}`$ | MMB$`^\text{D}`$ | SEED | MMMU$`^\text{V}`$ | MMMU$`^\text{T}`$ | VQA$`^\text{v2}`$ | GQA  | SQA$`^\text{I}`$ | POPE |
| ------------------------------------------------------------ | :--------------: | :--------------: | :--------------: | :--------------: | :--: | :---------------: | :---------------: | :---------------: | :--: | :----------------: | :----: |
| [bunny-phi-1.5-eva-lora](https://huggingface.co/BoyaWu10/bunny-phi-1.5-eva-lora) |      1213.7      |      278.9      |       60.9       |       56.8       | 56.4 | 30.0 |       28.4       |       76.5       | 60.4 | 58.2           | 86.1 |
| [bunny-stablelm-2-eva-lora](https://huggingface.co/BoyaWu10/bunny-stablelm-2-eva-lora) |      1301.0      |      235.0       |       58.4       |       56.4       | 55.3 | 29.8 |       29.4        |       74.6        | 56.7 | 60.0             | 84.8 |
| [bunny-phi-2-eva-lora](https://huggingface.co/BoyaWu10/bunny-phi-2-eva-lora) |      1421.0      |      285.4      |       68.6       |       67.4       | 62.2 | 35.9 |       32.6       |       78.9       | 62.3 | 69.1           | 87.1 |
| [bunny-phi-1.5-siglip-lora](https://huggingface.co/BoyaWu10/bunny-phi-1.5-siglip-lora) |      1230.0      |      237.5      |       61.2       |       59.7       | 57.7 | 30.0 |       29.1       |       78.0       | 61.1 | 61.3            | 85.8 |
| [bunny-stablelm-2-siglip-lora](https://huggingface.co/BoyaWu10/bunny-stablelm-2-siglip-lora) |      1366.8      |      236.1       |       65.1       |       62.8       | 58.8 | 29.9 |       29.8        |       78.9        | 60.9 | 61.1             | 85.9 |
| **[bunny-phi-2-siglip-lora](https://huggingface.co/BAAI/bunny-phi-2-siglip-lora)** |      1488.8      |      289.3      |       69.2       |       68.6       | 62.5 | 38.2 |       33.0       |       79.8       | 62.5 | 70.9        | 86.8 |

* Training details
  
| Checkpoint                                                   | Vision Encoder                                               | LLM                                                          | Pretrain lr | Pretrain weights                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :---------: | ------------------------------------------------------------ |
| [bunny-phi-1.5-eva-lora](https://huggingface.co/BoyaWu10/bunny-phi-1.5-eva-lora) | [EVA02_CLIP_L_336_psz14_s6B](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt) | [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) |    1e-3     | [bunny-pretrain-phi-1.5-eva](https://huggingface.co/BoyaWu10/bunny-pretrain-phi-1.5-eva) |
| [bunny-stablelm-2-eva-lora](https://huggingface.co/BoyaWu10/bunny-stablelm-2-eva-lora) | [EVA02_CLIP_L_336_psz14_s6B](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt) | [stabilityai/stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b) |    1e-3     | [bunny-pretrain-stablelm-2-eva](https://huggingface.co/BoyaWu10/bunny-pretrain-stablelm-2-eva) |
| [bunny-phi-2-eva-lora](https://huggingface.co/BoyaWu10/bunny-phi-2-eva-lora) | [EVA02_CLIP_L_336_psz14_s6B](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt) | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)    |    5e-5     | [bunny-pretrain-phi-2-eva](https://huggingface.co/BoyaWu10/bunny-pretrain-phi-2-eva) |
| [bunny-phi-1.5-siglip-lora](https://huggingface.co/BoyaWu10/bunny-phi-1.5-siglip-lora) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) |    5e-4     | [bunny-pretrain-phi-1.5-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-phi-1.5-siglip) |
| [bunny-stablelm-2-siglip-lora](https://huggingface.co/BoyaWu10/bunny-stablelm-2-siglip-lora) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [stabilityai/stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b) |    5e-4     | [bunny-pretrain-stablelm-2-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-stablelm-2-siglip) |
| **[bunny-phi-2-siglip-lora](https://huggingface.co/BAAI/bunny-phi-2-siglip-lora)** | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)    |    5e-4     | [bunny-pretrain-phi-2-siglip](https://huggingface.co/BAAI/bunny-pretrain-phi-2-siglip) |

## Install

* CUDA and cuDNN

  We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: `docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`

* Create a conda virtual environment and activate it:

  ```shell
  conda create -n bunny python=3.10
  conda activate bunny
  ```

* Basic requirements

  ```shell
  pip install --upgrade pip  # enable PEP 660 support
  pip install transformers==4.36.2
  pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
  ```

* Install apex

  ```shell
  # https://github.com/NVIDIA/apex#from-source
  pip install ninja
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

* Install flash-attention

  ```shell
  # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```

* Install bunny and other requirements

  ```shell
  git clone https://github.com/BAAI-DCAI/Bunny.git
  cd Bunny
  pip install -e .
  ```

## Training

Bunny training consists of two stages: (1) pretrain stage: use data to connect a *frozen pretrained* vision encoder to a *frozen* LLM, and only the connector is trained; (2) visual instruction tuning stage: use data to teach the model to follow multimodal instructions, where the connector and learnable LLM parameters are updated.

Bunny is trained on 8 A100 GPUs. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `global_batch_size ` = `per_device_train_batch_size` $`\times`$ `gradient_accumulation_steps` $`\times`$ `num_gpus`.

### Support Models

Currently, we support several vision encoders and LLMs.

For vision encoders, we support CLIP, EVA-CLIP and SigLIP.

| Vision Encoders            | Download Link                                                |
| -------------------------- | ------------------------------------------------------------ |
| clip-vit-large-patch14-336 | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) |
| EVA02_CLIP_L_336_psz14_s6B | [QuanSun/EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt) |
| siglip-so400m-patch14-384  | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |

For LLMs, we support phi-1.5, stablelm-2 and phi-2.

| MODEL_TYPE | LLM             | Download Link                                                |
| ---------- | --------------- | ------------------------------------------------------------ |
| phi-1.5    | phi-1_5     | [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) |
| stablelm-2 | stablelm-2-1_6b | [stabilityai/stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b) |
| phi-2      | phi-2           | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) |

Note that there are many variants of above models.
We build and test our code based on the exact versions mentioned above.
More models will be supported in the future!

### Pretrain

* Data preparation

  We use a high-quality coreset with less duplicates and more informative samples of LAION-2B built by [this work](https://github.com/BAAI-DCAI/Dataset-Pruning/tree/main/LAION). We randomly sample 2 million image-text pairs from the coreset and convert them to training format.
  The dataset will be released soon.

* Run

  Update `--model_name_or_path` and `--vision_tower` to the paths of the LLM and vision encoder, respectively. Update `MODEL_TYPE` and `OUTPUT_DIR` accordingly. The global batch size is 256. The optimal learning rate varies for different settings and we list the `lr` in our experiments in the [Model Zoo](#model-zoo).
  
  ```shell
  sh script/train/pretrain.sh
  ```

### Visual Instruction Tuning

* Data preparation

  We build Bunny-695K by modifying [SVIT-mix-665K](https://arxiv.org/abs/2307.04087) for finetuning.
  The dataset will be released soon.

* Run

  Update `--model_name_or_path` and `--vision_tower` to the paths of the LLM and vision encoder, respectively. Update `MODEL_TYPE`, `PRETRAIN_DIR` and `OUTPUT_DIR` accordingly. The global batch size is 128.
  
  ```shell
  # full-parameter tuning
  sh script/train/finetune_full.sh
  
  # LoRA tuning
  sh script/train/finetune_lora.sh
  ```

## Demo

### Gradio Web UI

* Starting the Controller

  First, start the controller. This service orchestrates communication between the web server and model workers.
  
  ```shell
  python -m bunny.serve.controller \
  	--host 0.0.0.0 \
  	--port 10000
  ```

* Launching the Gradio Web Server

  To interact with the models through a web interface, start the Gradio web server.

  Basic start:

  ```shell
  python -m bunny.serve.gradio_web_server \
  	--controller http://localhost:10000 \
  	--model-list-mode reload
  ```

  If you want to share your web server with others, use `--share` option.

  ```shell
  python -m bunny.serve.gradio_web_server \
  	--controller http://localhost:10000 \
  	--model-list-mode reload \
  	--share
  ```

  Now, you can open the web interface with **the URL printed on the screen**. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

* Launching Model Workers

  Model workers handle the processing of model inferences. Configure each worker with the appropriate model and start it.

  * For full-parameter tuning models

      ```shell
      python -m bunny.serve.model_worker \
        --host 0.0.0.0 \
        --controller http://localhost:10000 \
        --port 40000 \
        --worker http://localhost:40000 \
        --model-path /path/to/bunny/model \
        --model-type phi-2 (or stablelm-2 or phi-1.5)
      ```

  * For LoRA tuning models

      You can use `script/merge_lora_weights.py` to merge the LoRA weights and base LLM, and use it as above.
      
      ```Shell
      python script/merge_lora_weights.py \
        --model-path /path/to/bunny_lora_weights \
        --model-base /path/to/base_llm_model \
        --model-type phi-2 (or stablelm-2 or phi-1.5) \
        --save-model-path /path/to/merged_model
      ```
      Or you can use it without merging as below.
      
      ```shell
      python -m bunny.serve.model_worker \
        --host 0.0.0.0 \
        --controller http://localhost:10000 \
        --port 40000 \
        --worker http://localhost:40000 \
        --model-path /path/to/bunny_lora_weights \
        --model-base /path/to/base_llm_model \
        --model-type phi-2 (or stablelm-2 or phi-1.5)
      ```


### CLI Inference (Without Gradio Interface)

For CLI-based inference without using the Gradio interface, use the following command:

* For full-parameter tuning models

  ```shell
  python -m bunny.serve.cli \
  	--model-path /path/to/bunny/model \
  	--model-type phi-2 (or stablelm-2 or phi-1.5) \
  	--image-file /path/to/the/test/image
  ```

* For LoRA tuning models

  You can use `script/merge_lora_weights.py` to merge the LoRA weights and base LLM, and use it as above.

  ```Shell
  python script/merge_lora_weights.py \
  	--model-path /path/to/bunny_lora_weights \
  	--model-base /path/to/base_llm_model \
  	--model-type phi-2 (or stablelm-2 or phi-1.5) \
  	--save-model-path /path/to/merged_model
  ```

  Or you can use it without merging as below.

  ```shell
  python -m bunny.serve.cli \
  	--model-path /path/to/bunny_lora_weights \
  	--model-base /path/to/base_llm_model \
  	--model-type phi-2 (or stablelm-2 or phi-1.5) \
  	--image-file /path/to/the/test/image
  ```

## Evaluation

For full-parameter tuning models, see [evaluation_full.md](script/eval/full/evaluation_full.md).

For LoRA tuning models, see [evaluation_lora.md](script/eval/lora/evaluation_lora.md).

## License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).

## Acknowledgement

We build our project based on [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant.
