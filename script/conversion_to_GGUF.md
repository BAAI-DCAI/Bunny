# Conversion to GGUF

* Firstly, prepare a python environment and install the following dependencies:

  ```shell
  pip install torch transformers gguf sentencepiece
  ```

* And then install `llama.cpp`.

* Then, edit `llama.cpp/examples/llava/convert-image-encoder-to-gguf.py` to support SigLIP:

  * when importing packages, chage

    ```python
    from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
    ```

    to

    ```python
    from transformers import SiglipModel as CLIPModel
    from transformers import SiglipProcessor as CLIPProcessor
    from transformers import SiglipVisionModel as CLIPVisionModel
    ```


* Then, edit `llama.cpp/convert-hf-to-gguf.py` to skip unknown parts:

  change

  ```python
      def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
          new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
          if new_name is None:
              raise ValueError(f"Can not map tensor {name!r}")
          return new_name
  ```

  to

  ```python
      def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
          new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
          return new_name
  ```

  change

  ```python
      def write_tensors(self):
          max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")
  
          for name, data_torch in self.get_tensors():
              ...
  
              for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                  data: np.ndarray = data  # type hint
                  n_dims = len(data.shape)
                  ...
  ```

  to

  ```python
      def write_tensors(self):
          max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")
  
          for name, data_torch in self.get_tensors():
              ...
  
               for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                  if new_name is None:
                      continue
                    
                  data: np.ndarray = data  # type hint
                  n_dims = len(data.shape)
                  ...
  ```



## converting [Bunny-Llama-3-8B-V](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)

1. `cd llama.cpp/examples/llava`

2. Download the weights and put under `./`

3. Extract the weights of vision tower and multimodel projector:

   ```shell
   python llava-surgery-v2.py -C -m Bunny-Llama-3-8B-V
   ```

   you will find a `llava.projector` and a `llava.clip` file in `Bunny-Llama-3-8B-V`

4. Create the visual gguf model:

   * prepare files

     ```shell
     cd Bunny-Llama-3-8B-V
     mkdir vit
     cp llava.clip vit/pytorch_model.bin
     cp llava.projector vit/
     ```

     and put [`config.json`](#appendix) under `vit/`

   * and then:
   
     ```shell
     python ../convert-image-encoder-to-gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-vision --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
     cd ..
     ```
     
     you will find a `mmproj-model-f16.gguf` file in `Bunny-Llama-3-8B-V/vit`
     


5. Convert the left language part:

   * edit `Bunny-Llama-3-8B-V/config.json`:

     change

     ```json
       "architectures": [
         "BunnyLlamaForCausalLM"
       ],
       "auto_map": {
         "AutoConfig": "configuration_bunny_llama.BunnyLlamaConfig",
         "AutoModelForCausalLM": "modeling_bunny_llama.BunnyLlamaForCausalLM"
       },
     ```

     to

     ```json
       "architectures": [
         "LlamaForCausalLM"
       ],
     ```

   * And then:

     ```shell
     python ../../convert-hf-to-gguf.py Bunny-Llama-3-8B-V
     ```
     
     you will find a `ggml-model-f16.gguf` file in `Bunny-Llama-3-8B-V`
     
   



## converting [Bunny-v1_0-4B](https://huggingface.co/BAAI/Bunny-v1_0-4B)

1. `cd llama.cpp/examples/llava`

2. Download the weights and put under `./`

3. Extract the weights of vision tower and multimodel projector:

   ```shell
   python llava-surgery-v2.py -C -m Bunny-v1_0-4B
   ```

   you will find a `llava.projector` and a `llava.clip` file in `Bunny-v1_0-4B`

4. Create the visual gguf model:

   * prepare files

     ```shell
     cd Bunny-v1_0-4B
     mkdir vit
     cp llava.clip vit/pytorch_model.bin
     cp llava.projector vit/
     ```

     and put [`config.json`](#appendix) under `vit/`

   * and then:

     ```shell
     python ../convert-image-encoder-to-gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-vision --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
     cd ..
     ```
     
     you will find a `mmproj-model-f16.gguf` file in `Bunny-v1_0-4B/vit`
     


5. Convert the left language part:

   * edit `Bunny-v1_0-4B/config.json`:

     change

     ```json
       "architectures": [
         "BunnyPhi3ForCausalLM"
       ],
       "attention_dropout": 0.0,
       "auto_map": {
         "AutoConfig": "configuration_bunny_phi3.BunnyPhi3Config",
         "AutoModelForCausalLM": "modeling_bunny_phi3.BunnyPhi3ForCausalLM"
       },
     ```
   
     to
   
     ```json
       "architectures": [
         "Phi3ForCausalLM"
       ],
       "attention_dropout": 0.0,
       "auto_map": {
         "AutoConfig": "configuration_phi3.Phi3Config",
         "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"
       },
     ```
   
   * And then:
   
     ```shell
     python ../../convert-hf-to-gguf.py Bunny-v1_0-4B
     ```
   
     you will find a `ggml-model-f16.gguf` file in `Bunny-v1_0-4B`




## Appendix

`vit/config.json`

```json
{
  "architectures": [
    "SiglipVisionModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "freeze_mm_mlp_adapter": false,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_size": 1152,
  "image_size": 384,
  "image_aspect_ratio": "pad",
  "initializer_range": 0.02,
  "intermediate_size": 4304,
  "layer_norm_eps": 1e-6,
  "max_position_embeddings": 8192,
  "mm_hidden_size": 1152,
  "mm_projector_lr": 1e-05,
  "mm_projector_type": "mlp2x_gelu",
  "mm_vision_tower": "google/siglip-so400m-patch14-384",
  "model_type": "siglip_vision_model",
  "num_attention_heads": 16,
  "num_hidden_layers": 27,
  "num_key_value_heads": 8,
  "patch_size": 14,
  "pretraining_tp": 1,
  "projection_dim": 1152,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "tokenizer_model_max_length": 2048,
  "tokenizer_padding_side": "right",
  "torch_dtype": "float16",
  "transformers_version": "4.40.0",
  "tune_mm_mlp_adapter": false,
  "unfreeze_vision_tower": true,
  "use_cache": true,
  "use_mm_proj": true,
  "vocab_size": 128256
}
```



