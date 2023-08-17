import json



def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens


# return MB size
import os
from os.path import join, getsize

def get_dir_size(dir_path):
    dir_size = 0
    for root, dirs, files in os.walk(dir_path):
        dir_size += sum([getsize(join(root, name)) for name in files if 'dstore' not in name])
    return round(dir_size / 1024 / 1024, 2)




