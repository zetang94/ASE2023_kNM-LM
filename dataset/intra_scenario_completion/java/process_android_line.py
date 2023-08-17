import argparse
import json
import os
import codeprep.api.text as cp
import re
import javalang
from tqdm import tqdm

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


if __name__ == '__main__':
    samples = []
    with open('test_short.txt', 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            input_data = deal_with_code(line['source_code'], 'java')
            gt = deal_with_code(line['target'], 'java')
            api_sig = line['api_signature']
            if input_data is None or gt is None:
                continue
            sample = {'input': '<s> ' + input_data, 'gt': gt, 'api_signature': api_sig}
            samples.append(sample)

    save_file(samples, 'line_completion/Android/test.json', is_json=True)