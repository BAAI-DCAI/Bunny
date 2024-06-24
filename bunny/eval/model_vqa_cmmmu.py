import random
import numpy as np
import os
import json
import yaml
import torch

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser

from bunny.model.builder import load_pretrained_model
from bunny.util.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates

CAT_CN2EN = {'艺术与设计': 'art_and_design',
             '商业': 'business',
             '健康与医学': 'health_and_medicine',
             '人文社会科学': 'humanities_and_social_sciences',
             '科学': 'science',
             '技术与工程': 'technology_and_engineering'}


def call_bunny_engine_df(args, sample, model, tokenizer=None, processor=None):
    def deal_with_prompt(input_text):
        qs = input_text
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = sample['image_1']
    if sample['image_2'] is not None:  # multiple images actually
        if sample['type'] == '选择':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'
    elif image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
            do_sample=False,
            temperature=0,
            top_p=None,
            # num_beams=5,
            max_new_tokens=128,
            use_cache=True)

        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    return response


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


# DATA PROCESSING
def construct_prompt(sample, config):
    question = sample['question']
    options = []
    for i in range(1, 5):
        if sample[f'option{i}'] is None:
            break
        options.append(sample[f'option{i}'])

    example = ""
    if sample['type'] == '选择':
        start_chr = 'A'
        prediction_range = []
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][0].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = sample['answer']
    elif sample['type'] == '判断':
        empty_prompt_sample_structure = config['T/F_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][1].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][2].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = []
    with torch.no_grad():
        for sample in tqdm(samples):
            if args.small_gpu_usage:
                sample['image_1'] = sample['image_1'].cuda()
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)
            if args.small_gpu_usage:
                sample['image_1'] = sample['image_1'].cpu()

            out_sample = dict()
            out_sample['id'] = sample['id']
            out_sample['type'] = sample['type']
            out_sample['response'] = response
            out_samples.append(out_sample)
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--small-gpu-usage", action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('bunny_initializing...')
    processor = None
    call_model_engine = call_bunny_engine_df

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key == 'task_instructions':
            args.config[key] = value
        elif key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_CN2EN.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, vis_processors, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                          args.model_type)

    samples = []
    print('Processing CMMMU dataset...')
    for sample in tqdm(dataset):

        sample = construct_prompt(sample, args.config)
        if sample['image_1']:
            sample['image_1'] = process_images([sample['image_1'].convert('RGB')], vis_processors, model.config)[0]
            if not args.small_gpu_usage:
                sample['image_1'] = sample['image_1'].to(device)

        samples.append(sample)

    print('Start to evaluate...')
    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w') as f:
        for out_sample in out_samples:
            f.write(json.dumps(out_sample) + '\n')


if __name__ == '__main__':
    main()
