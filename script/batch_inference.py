import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

model_name = 'BAAI/Bunny-v1_1-Llama-3-8B-V'  # or 'BAAI/Bunny-Llama-3-8B-V' or 'BAAI/Bunny-v1_1-4B' or 'BAAI/Bunny-v1_0-4B' or 'BAAI/Bunny-v1_0-3B' or 'BAAI/Bunny-v1_0-3B-zh' or 'BAAI/Bunny-v1_0-2B-zh'

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# for batch inference
tokenizer.padding_side = "left"
tokenizer.pad_token_id = model.generation_config.pad_token_id
padding_max_length = 128  # customize for your circumstance
tokenizer.add_tokens(['<image>'])
image_token_id = tokenizer.convert_tokens_to_ids('<image>')

# text prompts
prompts = [
    'What is the astronaut holding in his hand?',
    'Why is the image funny?',
    'What is the occupation of the person in the picture?',
    'What animal is in the picture?'
]
texts = [
    f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
    for prompt in prompts]
input_ids = torch.tensor(
    [tokenizer(text, padding='max_length', max_length=padding_max_length).input_ids for text in texts],
    dtype=torch.long).to(device)
input_ids[input_ids == image_token_id] = -200

# images, sample images can be found in https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V/tree/main/images
image_paths = [
    'example_1.png',
    'example_2.png',
    'example_1.png',
    'example_2.png'
]
images = [Image.open(image_path) for image_path in image_paths]
image_tensor = model.process_images(images, model.config).to(dtype=model.dtype, device=device)

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True,
    repetition_penalty=1.0  # increase this to avoid chattering
)

print([ans.strip() for ans in tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)])
