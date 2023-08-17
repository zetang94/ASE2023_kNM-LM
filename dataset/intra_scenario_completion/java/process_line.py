import argparse
import json
import os
import codeprep.api.text as cp
import re
import javalang

lits = json.load(open("literals.json"))


def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    use_char = False
    if len(str_lit) == 1 and start_quote == "'":
        use_char = True
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    if not use_char:
        ret = (
            f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['str']
            else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
        )
    else:
        ret = (
            f"{qualifier}{start_quote}<CHAR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['char']
            else f"{qualifier}{start_quote}<CHAR_LIT>{end_quote}"
        )
    return ret


def deal_with_code(code, lang):
    no_spaces = True if lang == 'java' else False
    tokens = cp.nosplit(code,
                        extension=lang,
                        no_spaces=no_spaces,
                        no_unicode=True,
                        no_com=True,
                        full_strings=True,
                        max_str_length=15,
                        return_metadata=False)
    tokens = ' '.join([t for t in tokens if t != '<comment>'])
    new_data = []
    try:
        for tok in list(javalang.tokenizer.tokenize(tokens)):
            if "String" in str(type(tok)) or "Character" in str(type(tok)):
                token = process_string(tok.value)
            elif "Integer" in str(type(tok)) or "FloatingPoint" in str(type(tok)):
                if tok.value in lits['num']:
                    token = f"<NUM_LIT:{tok.value}>"
                else:
                    token = "<NUM_LIT>"
            else:
                token = tok.value
            new_data.append(token)
    except Exception as e:
        return None
    if len(new_data) == 0:
        return None

    data = " ".join(new_data)
    return data


def parse_directory(dir_path):
    if not os.path.exists(dir_path):
        print(f'Could not find directory: {dir_path}')

    walk = os.walk(dir_path)
    contents = []
    for sub_dir, _, files in walk:
        for filename in files:
            path = os.path.join(sub_dir, filename)

            if filename.endswith('.java'):
                #print('path: ', path)
                file_content = parse_file(path)
                if file_content is not None:
                    contents.append(file_content)
                    if len(contents) % 100 == 0:
                        print(len(contents), ' are done!')

    return contents


def parse_file(path):
    code = ''
    with open(path, 'r', encoding='utf-8') as file:
        code = file.read()

    code_tokens = deal_with_code(code, 'java')

    if code_tokens is not None:
        return "<s> " + code_tokens + " </s>"
    else:
        return None


def process_token_dataset(data_dir, output_dir):
    print('process token completion dataset.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #for file_type in ['Training', 'Testing']:
    for file_type in ['Training', 'Testing']:
        new_data_dir = os.path.join(data_dir, file_type)
        results = parse_directory(new_data_dir)
        if file_type == 'Training':
            dev_len = len(results) // 10
            train_len = len(results) - dev_len
            dev_results = results[-dev_len:]
            train_results = results[: train_len]

            save_file(train_results, output_dir + '/train.txt')
            save_file(dev_results, output_dir + '/dev.txt')

        else:
            save_file(results, output_dir + '/test.txt')


def save_file(samples, file_name, is_json=False):
    with open(file_name, 'w') as f:
        if is_json:
            for i, s in enumerate(samples):
                if i < len(samples) - 1:
                    f.write(json.dumps(s) + '\n')
                else:
                    f.write(json.dumps(s))
        else:
            for i, s in enumerate(samples):
                if i < len(samples) - 1:
                    f.write(s + '\n')
                else:
                    f.write(s)


def process_line_completion(data_dir, domain, extract_type, output_dir):
    print('process line completion dataset.')
    types = ['long', 'normal', 'short']
    idx = 0

    if not os.path.exists(f"{output_dir}/{domain}"):
        os.makedirs(f"{output_dir}/{domain}")

    samples = []

    for t in types:
        meta_dir = f'{data_dir}/Java/metadata_len/{domain}_{t}_withMetaData.json'

        meta_data = json.load(open(meta_dir, 'r'))
        for file_path in meta_data.keys():
            if not file_path.startswith(f'CodeDataset/Java/{domain}/Testing'):
                continue
            code = open(os.path.join(data_dir, file_path)).read()
            code = code.split('\n')

            for class_name in meta_data[file_path]:
                for func_name in meta_data[file_path][class_name]:

                    used = False
                    for api_call, detail in meta_data[file_path][class_name][func_name].items():
                        lib_name = detail[-1]
                        # if api_call == '@FuncLoc@':
                        #     start_pos = detail[0] - 1
                        if api_call != '@FuncLoc@' and lib_name != 'Unknown':
                            if extract_type == 'domain' and lib_name in ['Standard', 'User-defined']:
                                continue
                            if extract_type == 'Standard' and lib_name != 'Standard':
                                continue
                            if extract_type == 'User-defined' and lib_name != 'User-defined':
                                continue

                            # 同一个方法只取一处
                            loc = detail[0]
                            if isinstance(loc, list):
                                new_input_data = '\n'.join(code[:loc[0] - 1])
                                input_data = deal_with_code(new_input_data, 'java')
                                gt = deal_with_code(code[loc[0] - 1], 'java')
                                if input_data is None or gt is None:
                                    continue

                                sample = {'input': '<s> ' + input_data, 'gt': gt,
                                          'file_path': file_path}
                                if not used:
                                    samples.append(sample)
                                    used = True

                                if idx % 1000 == 0:
                                    print(f'{idx} is done!')

                                idx += 1

    save_file(samples, f'{output_dir}/{domain}/test.json', is_json=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="", type=str,
                        help="domain dataset dir")
    args = parser.parse_args()

    domains = ['Android']

    for d in domains:
        process_line_completion(data_dir=args.data_dir,
                                domain=d,
                                extract_type='domain',
                                output_dir='./line_completion')

        # process_token_dataset(data_dir=args.data_dir + '/CodeDataset/Java/' + d,
        #                       output_dir='./token_completion/' + d)

        print(f'{d} data is processed.')