import json
import git
import os.path
from shutil import copytree
from typing import List, Dict
import os
from tree_sitter import Language, Parser
from tqdm import tqdm

from utils import deal_with_code

language_library = Language('./build/my-languages.so', 'java')
parser = Parser()
parser.set_language(language_library)
method_q = language_library.query("""
            (method_declaration
                name: (identifier) @method_name
                parameters: (formal_parameters) @method_params) @method
            (constructor_declaration
                name: (identifier) @method_name
                parameters: (formal_parameters) @method_params) @method
            """)


def node_to_string(lines, node) -> str:
    start_point = node.start_point
    end_point = node.end_point
    if start_point[0] == end_point[0]:
        return lines[start_point[0]][start_point[1]:end_point[1]]
    ret = lines[start_point[0]][start_point[1]:] + "\n"
    ret += "\n".join([line for line in lines[start_point[0] + 1:end_point[0]]])
    ret += "\n" + lines[end_point[0]][:end_point[1]]
    return ret


def write_methods(method_list: List[Dict[str, str]], output_dir: str, file_type: str) -> None:
    with open(os.path.join(output_dir, f"{file_type}.txt"), "w") as f:
        for i in range(len(method_list)):
            f.write(method_list[i]["code"] + "\n")


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
                    contents.extend(file_content)
                    if len(contents) % 100 == 0:
                        print(len(contents), ' are done!')

    return contents


def parse_file(path):
    code = ''
    with open(path, 'r', encoding='utf-8') as file:
        code = file.read()
        lines = code.split('\n')

    try:
        tree = parser.parse(bytes(code, "utf8"))
        captures = method_q.captures(tree.root_node)
        cur_method_nodes = [n for n, i in enumerate(captures) if i[1] == 'method']

        methods = [node_to_string(lines, captures[node][0]) for node in cur_method_nodes]

        contents = parse_methods(methods)

        return contents
    except Exception as e:
        print(e)
        return None


def parse_methods(methods: List) -> List:
    contents = []
    for method in methods:
        content = {}
        try:
            code_tokens, code_types = deal_with_code(method, 'java')

            data = ['<s>'] + code_tokens + ["</s>"]
            data_type = ['tag'] + code_types + ['tag']
            content['code'] = data
            content['token_type'] = data_type

            contents.append(content)
        except Exception:
            continue

    return contents


def write_to_file(contents, output_dir, file_type):
    wf = open(os.path.join(output_dir, f"{file_type}.txt"), 'w')
    wf_type = open(os.path.join(output_dir, f"{file_type}_type.txt"), 'w')
    for content in tqdm(contents, desc='write to file ' + file_type + '...'):
        wf.write(json.dumps(content['code']) + "\n")
        wf_type.write(json.dumps(content['token_type']) + "\n")

    wf.flush()
    wf.close()
    wf_type.flush()
    wf_type.close()


def filter_duplicates(methods: List) -> List:
    raw_codes = set()
    filtered = []
    for method in methods:
        clear_code = "".join(method["code"].split())
        if clear_code not in raw_codes:
            filtered.append(method)
            raw_codes.add(clear_code)
    return filtered


def split_dataset(project_name: str, train_part: float, data_dir: str, repo_dir: str) -> str:
    raw_samples = open(os.path.join(data_dir, f"{project_name}.jsonl"), "r")
    added_methods = [sample for sample in list(map(json.loads, raw_samples)) if sample["update"] == "ADD"]
    added_methods.sort(key=lambda method: method["commitTime"])
    added_methods = filter_duplicates(added_methods)
    num_of_methods = len(added_methods)

    start_idx = int(train_part * num_of_methods) - 1
    snapshot_commit = added_methods[start_idx]["commitId"]

    last_commit = added_methods[-1]["commitId"]
    print("*"*10)

    print("Repo name: ", project_name)
    print("start commit: ", snapshot_commit)
    print("last commit: ", last_commit)

    print("*"*10)

    source_dir = os.path.join(repo_dir, project_name)

    repo = git.Repo(source_dir)
    repo.head.reset(snapshot_commit, index=True, working_tree=True)

    train_path = os.path.join(data_dir, f"tmp/{project_name}")

    copytree(source_dir, train_path)

    """ensure that train, valid, test do not have the same commit files."""
    new_idx = start_idx + 1
    for idx in range(start_idx, num_of_methods):
        if added_methods[idx]["commitId"] != snapshot_commit:
            new_idx = idx
            break

    num_of_val_methods = (num_of_methods - new_idx) // 2
    #train_methods = added_methods[0: new_idx]
    val_methods = [s['code'] for s in added_methods[new_idx: new_idx + num_of_val_methods]]
    test_methods = [s['code'] for s in added_methods[new_idx + num_of_val_methods:]]

    output_dir = os.path.join(data_dir, f"output/{project_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_methods = parse_directory(train_path)
    write_to_file(train_methods, output_dir, 'train')

    val_methods = parse_methods(val_methods)
    write_to_file(val_methods, output_dir, 'dev')

    test_methods = parse_methods(test_methods)
    write_to_file(test_methods, output_dir, 'test')

    return 'done.'


def exit_with_message(message):
    print(f"{message} Exiting...")
    exit(1)


if __name__ == '__main__':
    dir_path = "/Volumes/T7 Shield/projects/jetbrains_dataset/"
    project_names = os.listdir(dir_path + 'repos')
    for project_name in project_names:
        if project_name.startswith('.DS') or project_name == "._.DS_Store" or project_name == 'MSEC':
            continue
        print(project_name)

        output_dir = os.path.join(dir_path + 'outputs', f"output/{project_name}")
        if os.path.exists(output_dir):
            continue

        split_dataset(project_name, 0.8,
                      dir_path + 'outputs',
                      dir_path + 'repos')





