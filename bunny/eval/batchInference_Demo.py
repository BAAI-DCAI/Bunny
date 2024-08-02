import json
import os
import copy
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import torch

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

device = 'cuda'
torch.set_default_device(device)

tokenizer = AutoTokenizer.from_pretrained(
    'BAAI/Bunny-Llama-3-8B-V',
    trust_remote_code=True, model_max_length=512)
tokenizer.eos_token_id = 128001
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained('BAAI/Bunny-Llama-3-8B-V', torch_dtype=torch.float16, device_map='auto',
                                             trust_remote_code=True, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,)

list_data_dict = json.load(open('../../bunny/data/finetune/bunny_695k.json', "r"))
list_data_dict = sorted(list_data_dict, key=lambda x: sum(len(conv['value']) for conv in x['conversations']))
print("sort done.")
with open('../../bunny/data/finetune/bunny_695k_sorted.json', 'w') as f:
    json.dump(list_data_dict, f, ensure_ascii=False, indent=4)

image_folder = '../../bunny/data/finetune/images'
output_folder = './bunny_695k_updated/'
output_folder_ori = './bunny_695k_original/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_ori, exist_ok=True)
def replace_gpt_value(conversations, new_value):
    for conversation in conversations:
        if conversation['from'] == 'gpt':
            conversation['value'] = new_value
    return conversations

def prepare_batch_text_and_images(indices):
    """Prepare batch text and image tensors for the given indices."""
    text_batches, image_batches, ori_convs, has_image_list, chunk_batches = [], [], [], [], []
    for idx in indices:
        ori_conv_cur = []
        item = list_data_dict[idx]
        if 'image' in item:
            image_path = os.path.join(image_folder, item['image'])
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
                has_image_list.append(1)
            else:
                image_tensor = torch.zeros((1, 3, 384, 384), dtype=model.dtype, device=device)
                has_image_list.append(0)
        else:
            image_tensor = torch.zeros((1, 3, 384, 384), dtype=model.dtype, device=device)
            has_image_list.append(0)
        image_placeholder = "<image>\n" if image_tensor is not None else ""
        text_pairs = []
        chunk_input_ids = []
        for conv in item['conversations']:
            if conv['from'] == 'human':
                flag = 0
                question = f"{conv['value']}".replace('<image>\n', '')
            if conv['from'] == 'gpt':
                flag = 1
                answer = f"Reference Answer:\n{conv['value']}".replace('<image>\n', '')
            if flag == 1:
                flag = 0
                text = f"User:{image_placeholder}{question}\nAssistant:"
                text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
                input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
                chunk_input_ids.append(input_ids)
                
                text_pairs.append(text)
        ori_conv_q = item['conversations'][0]
        ori_conv_a = item['conversations'][1]
        ori_conv_cur.append(ori_conv_q)
        ori_conv_cur.append(ori_conv_a) 

        text_batches.append(text_pairs[0])
        image_batches.append(image_tensor)
        ori_convs.append(ori_conv_cur)
        chunk_batches.append(chunk_input_ids[0])

    constant_value = 128001
    max_len = max(tensor.shape[1] for tensor in chunk_batches)
    chunk_batches_final = []
    for tensor in chunk_batches:
        pad_len = max_len - tensor.shape[1]
        padded_tensor = torch.cat([torch.full((tensor.shape[0], pad_len), constant_value, device=tensor.device), tensor], dim=1)
        chunk_batches_final.append(padded_tensor)
    chunk_batches_final = torch.cat(chunk_batches_final, dim=0)
    return text_batches, image_batches, ori_convs, has_image_list, chunk_batches_final


new_data_list = []
print("total data num: ", len(list_data_dict))
batch_size = 4
num_batches = len(list_data_dict) // batch_size
print("total batches: ", num_batches)
all_new_data = []
all_ori_data = []
save_batch = 6250
cur = 1
for i in range(num_batches):
    print("current batch: ", i)
    print("\n")
    indices = list(range(i * batch_size, (i + 1) * batch_size))
    text_batches, image_batches, ori_convs_batches, has_image_list, chunk_batches = prepare_batch_text_and_images(indices)

    inputs = tokenizer(text_batches, return_tensors='pt', padding="longest",
                       max_length=tokenizer.model_max_length,
                       truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    input_length = input_ids.size(1)
    output_ids = model.generate(chunk_batches, images=image_batches, max_new_tokens=512, use_cache=True,
                                repetition_penalty=1.0, temperature=0.0, do_sample=False)

    output_ids = [output_id[input_length:] for output_id in output_ids]
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    gen_convs_batches = copy.deepcopy(ori_convs_batches)
    for mm in range(len(indices)):
        generated_text = generated_texts[mm]
        replace_gpt_value(gen_convs_batches[mm], generated_text)
        gen_convs = gen_convs_batches[mm]
        ori_convs = ori_convs_batches[mm]
        if has_image_list[mm] == 1:
            new_data_dict = {
                'id': list_data_dict[indices[mm]]['id'],
                'image': list_data_dict[indices[mm]]['image'],
                'conversations': gen_convs_batches[mm]
            }
            ori_data_dict = {
                'id': list_data_dict[indices[mm]]['id'],
                'image': list_data_dict[indices[mm]]['image'],
                'conversations': ori_convs_batches[mm]
            }
            print("new_data_dict (inference): \n", new_data_dict)
            print("ori_data_dict (ground truth): \n", ori_data_dict)
        else:
            new_data_dict = {
                'id': list_data_dict[indices[mm]]['id'],
                'conversations': gen_convs_batches[mm]
            }
            ori_data_dict = {
                'id': list_data_dict[indices[mm]]['id'],
                'conversations': ori_convs_batches[mm]
            }
            print("new_data_dict (inference): \n", new_data_dict)
            print("ori_data_dict (ground truth): \n", ori_data_dict)
        all_new_data.append(new_data_dict)
        all_ori_data.append(ori_data_dict)
    if (i+1)%save_batch == 0:
        new_data_filename = os.path.join(output_folder, 'bunny_695k_gen_bs{}_part{}.json'.format(batch_size, cur))
        with open(new_data_filename, 'w') as new_data_file:
            json.dump(all_new_data, new_data_file, ensure_ascii=False, indent=4)
        ori_data_filename = os.path.join(output_folder_ori, 'bunny_695k_ori_bs{}_part{}.json'.format(batch_size, cur))
        with open(ori_data_filename, 'w') as ori_data_file:
            json.dump(all_ori_data, ori_data_file, ensure_ascii=False, indent=4)
        cur += 1
        all_new_data = []
        all_ori_data = []
