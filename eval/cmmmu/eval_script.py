import json
from argparse import ArgumentParser
from tabulate import tabulate
from eval_utils import evaluate_answer, evaluate_response


def read_jsonl_to_dict(data_path, output_path, category):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = {int(parsed_line['id']): parsed_line for line in file if
                (parsed_line := json.loads(line)).get('category') == category}

    with open(output_path, 'r', encoding='utf-8') as file:
        output = {int(parsed_line['id']): parsed_line for line in file if
                  int((parsed_line := json.loads(line)).get('id')) in data.keys()}

    return data, output


def process_answer_jsonl_file(data_path, output_path, category):
    global global_cnt
    global global_correct_cnt

    data_dict, output_dict = read_jsonl_to_dict(data_path, output_path, category)

    assert set(data_dict.keys()) == set(
        output_dict.keys()), "The ids are not exactly the same and cannot be processed further, please check files"

    for data_key, data_value in data_dict.items():
        output_dict[data_key]['predicted_answer'] = output_dict[data_key].get('answer')
        output_dict[data_key]['answer'] = data_value.get('answer')

    results_count = evaluate_answer(output_dict.values())

    return results_count


def process_response_jsonl_file(data_path, output_path, category):
    global global_cnt
    global global_correct_cnt

    data_dict, output_dict = read_jsonl_to_dict(data_path, output_path, category)

    assert set(data_dict.keys()) == set(
        output_dict.keys()), "The ids are not exactly the same and cannot be processed further, please check files"

    for data_key, data_value in data_dict.items():
        if data_value.get('type') == "选择":
            index2ans = {
                'A': data_value.get('option1', ''),
                'B': data_value.get('option2', ''),
                'C': data_value.get('option3', ''),
                'D': data_value.get('option4', '')
            }
            output_dict[data_key]['index2ans'] = index2ans
        output_dict[data_key]['answer'] = data_value.get('answer')

    results_count = evaluate_response(output_dict.values())

    return results_count


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default="eval/example/Yi-VL-34B-answer.jsonl",
                        help="The path to model output file.")
    parser.add_argument('--data_path', type=str, default="eval/cmmmu/cmmmu-data-val-answer.jsonl",
                        help="Answer file path.")
    args = parser.parse_args()

    category_list = ['艺术与设计', '商业', '科学', '健康与医学', '人文社会科学', '技术与工程']
    category_dict = {'艺术与设计': 'Art & Design', '商业': 'Business', '科学': 'Science',
                     '健康与医学': 'Health & Medicine', '人文社会科学': 'Humanities & Social Sciences',
                     '技术与工程': 'Technology & Engineering'}

    headers = ['Subject', 'Correct Num', 'Entries Num', 'Acc']
    table = []
    correct_sum = 0
    entries_sum = 0

    is_answer = True
    is_response = True
    with open(args.output_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if set(data.keys()) != {'id', 'type', 'answer'}:
                is_answer = False
            if set(data.keys()) != {'id', 'type', 'response'}:
                is_response = False
    assert is_answer or is_response, "The file should contain either 'answer' or 'response'"

    for category in category_list:
        if is_answer:
            results_count = process_answer_jsonl_file(args.data_path, args.output_path, category)
        elif is_response:
            results_count = process_response_jsonl_file(args.data_path, args.output_path, category)
        correct_sum += results_count['correct_num']
        entries_sum += results_count['entries_num']
        table.append(
            [category_dict[category], results_count['correct_num'], results_count['entries_num'], results_count['acc']])

    table.append(['Overall', correct_sum, entries_sum, correct_sum / entries_sum])
    print(tabulate(table, headers=headers, tablefmt='orgtbl'))
