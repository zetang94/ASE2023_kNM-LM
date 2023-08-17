import argparse
import json
import os.path

import javalang
from tqdm import tqdm

from utils import deal_with_code


def parse_directory(dir_path):
    if not os.path.exists(dir_path):
        exit_with_message(f'Could not find directory: {dir_path}')

    walk = os.walk(dir_path)
    contents = []
    for sub_dir, _, files in walk:
        for filename in files:
            path = os.path.join(sub_dir, filename)
            if filename.endswith('.java'):
                file_content = parse_file(path)
                if file_content is not None:
                    contents.append(file_content)
                    if len(contents) % 100 == 0:
                        print(len(contents), ' are done!')

    return contents


def write_to_file(contents, file_type, output_dir):
    wf = open(os.path.join(output_dir, f"{file_type}.txt"), 'w')
    wf_type = open(os.path.join(output_dir, f"{file_type}_type.txt"), 'w')
    for content in tqdm(contents, desc='write to file ' + file_type + '...'):
        wf.write(json.dumps(content['code']) + "\n")
        wf_type.write(json.dumps(content['token_type']) + "\n")

    wf.flush()
    wf.close()
    wf_type.flush()
    wf_type.close()


def parse_file(path):
    code = ''
    with open(path, 'r', encoding='utf-8') as file:
        code = file.read()

    content = {}
    try:
        code_tokens, code_types = deal_with_code(code, 'java')

        data = ['<s>'] + code_tokens + ["</s>"]
        data_type = ['tag'] + code_types + ['tag']
        content['code'] = data
        content['token_type'] = data_type

        return content
    except Exception:
        return None


def exit_with_message(message):
    print(f"{message} Exiting...")
    exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="", type=str,
                        help="domain dataset dir")
    parser.add_argument("--output_dir", default="token_completion", type=str,
                        help="The output directory")
    args = parser.parse_args()

    domains = ['Android', 'ML', 'Security', 'Test']

    for d in domains:
        output_dir = os.path.join(args.output_dir, d)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_dir = os.path.join(args.data_dir, d)
        # need to split train and dev dataset!
        for file_type in ['Training', 'Testing']:
            data_dir = os.path.join(base_dir, file_type)
            results = parse_directory(data_dir)

            if file_type == 'Training':
                dev_len = len(results) // 10
                train_len = len(results) - dev_len
                dev_results = results[-dev_len:]
                train_results = results[: train_len]

                write_to_file(train_results, 'train', output_dir)
                write_to_file(dev_results, 'dev', output_dir)

            else:
                write_to_file(results, 'test', output_dir)


