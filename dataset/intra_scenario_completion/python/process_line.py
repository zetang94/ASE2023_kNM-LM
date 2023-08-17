import argparse
import json
import os
from numbers import Number
import codeprep.api.text as cp
import re
from codeprep.tokens.containers import SplitContainer, StringLiteral, TextContainer
from codeprep.tokens.whitespace import NewLine, Tab
from codeprep.tokens.word import KeyWord, Semicolon, OpeningCurlyBracket, OpeningBracket, ClosingBracket, Operator, \
    ClosingCurlyBracket

import json
lits = json.load(open("literals.json"))

keywords = [KeyWord.__name__]

punctuations = [Semicolon.__name__, OpeningCurlyBracket.__name__,
                OpeningBracket.__name__, ClosingBracket.__name__,
                ClosingCurlyBracket.__name__]

py_punct = [NewLine.__name__, Tab.__name__]

operators = [Operator.__name__]

identifiers = [SplitContainer.__name__]

literals = [StringLiteral.__name__, Number.__name__]


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
            f"<STR_LIT:{str_lit}>"
            if str_lit in lits['str']
            else f"<STR_LIT>"
        )
    else:
        ret = (
            f"<CHAR_LIT:{str_lit}>"
            if str_lit in lits['char']
            else f"<CHAR_LIT>"
        )
    return ret


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


def deal_with_code(code, lang):
    no_spaces = True if lang == 'java' else False
    tokens, metadata = cp.nosplit(code,
                                  extension=lang,
                                  no_spaces=no_spaces,
                                  no_unicode=True,
                                  no_com=True,
                                  full_strings=True,
                                  max_str_length=15,
                                  return_metadata=True)

    token_types = list(map(lambda x: x.__name__, metadata.token_types))

    processed_tokens = []
    for i, (token, token_type) in enumerate(zip(tokens, token_types)):
        if token == '<comment>':
            continue
        # deal with string
        if token_type == StringLiteral.__name__:
            # processed_tokens.append(token)
            if token in ["'''", '"""', "'", '"']:
                pass
            else:
                token = process_string(token)

        # deal with number
        if token_type == Number.__name__:
            if token in lits['num']:
                token = f"<NUM_LIT:{token}>"
            else:
                token = "<NUM_LIT>"

        # for python
        if token_type in py_punct:
            token = '<EOL>'
            if i - 1 >= 0 and token_types[i - 1] in py_punct:
                continue

        if ' ' in token:
            sub_token = token.split()
            processed_tokens.extend(sub_token)
        else:
            processed_tokens.append(token)

    if len(processed_tokens) == 0:
        return None

    if processed_tokens[0] == '<EOL>':
        processed_tokens = processed_tokens[1:]

    if  len(processed_tokens) > 0 and processed_tokens[-1] == '<EOL>':
        processed_tokens = processed_tokens[:-1]

    if len(processed_tokens) == 0:
        return None

    data = " ".join(processed_tokens)
    return data


def process_line_completion(data_dir, domain, extract_type, output_dir):
    print('process line completion dataset.')
    types = ['long', 'normal', 'short']
    idx = 0

    if not os.path.exists(f"{output_dir}/{domain}"):
        os.makedirs(f"{output_dir}/{domain}")

    samples = []

    for t in types:
        meta_dir = f'{data_dir}/Python/metadata_len/{domain}_{t}_withMetaData.json'

        meta_data = json.load(open(meta_dir, 'r'))
        for file_path in meta_data.keys():
            if not file_path.startswith(f'CodeDataset/Python/{domain}/Testing'):
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
                                input_data = deal_with_code(new_input_data, 'py')
                                gt = deal_with_code(code[loc[0] - 1], 'py')
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


def parse_file(path):
    code = ''
    with open(path, 'r', encoding='utf-8') as file:
        code = file.read()

    code_tokens = deal_with_code(code, 'py')

    if code_tokens is not None:
        return "<s> " + code_tokens + " </s>"
    else:
        return None


def parse_directory(dir_path):
    if not os.path.exists(dir_path):
        print(f'Could not find directory: {dir_path}')

    walk = os.walk(dir_path)
    contents = []
    for sub_dir, _, files in walk:
        for filename in files:
            path = os.path.join(sub_dir, filename)

            if filename.endswith('.py'):
                #print('path: ', path)
                file_content = parse_file(path)
                if file_content is not None:
                    contents.append(file_content)
                    if len(contents) % 100 == 0:
                        print(len(contents), ' are done!')

    return contents


def process_token_dataset(data_dir, output_dir):
    print('process token completion dataset.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_type in ['Training', 'Testing']:
    #for file_type in ['Testing']:
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="", type=str,
                        help="domain dataset dir")
    args = parser.parse_args()

    domains = ['DL']

    for d in domains:
        process_line_completion(data_dir=args.data_dir,
                                domain=d,
                                extract_type='domain',
                                output_dir='./line_completion')

        # process_token_dataset(data_dir=args.data_dir + '/CodeDataset/Python/' + d,
        #                       output_dir='./token_completion/' + d)

        print(f'{d} data is processed.')