from __future__ import absolute_import, division, print_function

import os
import pickle
import gc
import logging
import json
import re
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)


def tokenize(source, tokenizer, mode_type):
    source = source.strip()

    # if source.startswith("<s>") and source.endswith("</s>"):
    #     if mode_type == 'unixCoder':
    #         source = "<s> <decoder-only> </s> " + source[4:]
    # else:
    #     if mode_type == 'unixCoder':
    #         source = "<s> <decoder-only> </s> " + source + " </s>"
    #     elif mode_type == 'gpt2':
    #        source = "<s> " + source + " </s>"
    if not source.startswith("<s>"):
        source = "<s> " + source + " </s>"
    source_tokens = [t for t in tokenizer.tokenize(source)]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    return source_ids


def build_token_completion_data(tokenizer, args, logger, file_type='train', block_size=1024):

    cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size))
    before_context_file = os.path.join(args.output_dir + '/datastore', file_type + "_query.json")

    datafile = os.path.join(args.data_dir, f"{file_type}.txt")
    with open(datafile) as f:
        data = f.readlines()

    length = len(data)
    logger.info("Data size: %d" % (length))
    # input_ids = []
    split_inputs = []
    before_contexts = []
    _sample_id = 0
    sep_id = tokenizer.convert_tokens_to_ids(['\u0120'])
    for idx, x in enumerate(data):
        try:
            x = json.loads(x)
            s = len([t for t in x if '<STR_LIT' in t])

            if s > 1024:
                continue

            x = ' '.join(x)
            ids = tokenize(x, tokenizer, args.model_type)
            max_chunk_len = block_size // 4
            i = 0
            while i < len(ids):
                sample = ids[i: i + block_size]
                if len(sample) == block_size:
                    for j in range(block_size):
                        if tokenizer.convert_ids_to_tokens(sample[block_size - 1 - j])[
                            0] == '\u0120' or tokenizer.convert_ids_to_tokens(
                            sample[block_size - 1 - j]).startswith("<NUM_LIT"):
                            break
                        if sample[block_size - 1 - j] in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                                                          tokenizer.sep_token_id]:
                            if sample[block_size - 1 - j] != tokenizer.bos_token_id:
                                j -= 1
                            break
                    if j == block_size - 1:
                        if file_type == 'train':
                            i = i + block_size
                            continue
                        print(tokenizer.decode(sample))
                        exit()
                    sample = sample[: block_size - 1 - j]

                    i += len(sample)
                    # pad_len = block_size - len(sample)
                    # sample += [tokenizer.pad_token_id] * pad_len
                    #
                    # input_ids.append(sample)
                else:
                    # pad_len = block_size - len(sample)
                    # sample += [tokenizer.pad_token_id] * pad_len
                    # sample = bos_ids + sample
                    # input_ids.append(sample)
                    i += len(sample)

                # 向上取整
                sub_len = math.ceil(len(sample) / 4)

                before_sub_sample = []

                for k in range(4):
                    begin_index = k * sub_len
                    end_index = (k + 1) * sub_len

                    if begin_index > len(sample):
                        continue

                    sub_sample = sample[begin_index: end_index]

                    # if sub_sample[-1] not in [tokenizer.eos_token_id,
                    #                           tokenizer.sep_token_id,
                    #                           sep_id]:
                    #     sub_sample += sep_id  # solve post process bug.

                    split_inputs.append(sub_sample)
                    if not len(before_sub_sample) == 0:
                        before_contexts.append({'id': _sample_id,
                                                'input': tokenizer.decode(before_sub_sample)})
                    _sample_id += 1
                    before_sub_sample = before_sub_sample + sub_sample
                    before_sub_sample = before_sub_sample[-max_chunk_len:]
        except Exception:
            pass
        if idx % (length // 10) == 0:
            percent = idx / (length // 10) * 10
            logger.warning("load %d" % (percent))

    del data
    gc.collect()

    # logger.info(f"tokens: {len(input_ids)}")
    # split_inputs, before_contexts = split_for_token_completion(input_ids, tokenizer, logger, model_type=args.model_type,
    #                                                            file_type=file_type,
    #                                                            block_size=block_size)
    # del input_ids
    #gc.collect()

    with open(cached_file, 'wb') as handle:
        pickle.dump(split_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(before_context_file, 'w') as f:
        for i, s in enumerate(before_contexts):
            if i < len(before_contexts) - 1:
                f.write(json.dumps(s) + '\n')
            else:
                f.write(json.dumps(s))


# def split_for_token_completion(input_ids, tokenizer, logger, model_type, file_type, block_size=1024):
#     idx = 0
#
#     split_inputs = []
#     before_contexts = []
#     i = 0
#
#     while i < len(input_ids):
#         sample = input_ids[i: i + block_size]
#         if len(sample) == block_size:
#             for j in range(block_size):
#                 if tokenizer.convert_ids_to_tokens(sample[block_size - 1 - j])[
#                     0] == '\u0120' or tokenizer.convert_ids_to_tokens(sample[block_size - 1 - j]).startswith(
#                         "<NUM_LIT"):
#                     break
#
#                 if sample[block_size - 1 - j] in [tokenizer.bos_token_id, tokenizer.eos_token_id,
#                                                   tokenizer.sep_token_id]:
#                     if sample[block_size - 1 - j] == tokenizer.eos_token_id:
#                         if model_type == 'unixCoder' and \
#                                 sample[block_size - 3 - j] == tokenizer.bos_token_id:
#                             j += 2
#                         else:
#                             j -= 1
#                     elif sample[block_size - 1 - j] == tokenizer.sep_token_id:
#                         j -= 1
#
#                     break
#             if j == block_size - 1:
#                 if file_type == 'dev':
#                     i = i + block_size
#                     continue
#                 print(tokenizer.decode(sample))
#                 exit()
#             sample = sample[: block_size - 1 - j]
#         # print(len(sample))
#         i += len(sample)
#
#         sub_len = block_size // 4
#
#         before_sub_sample = None
#
#         for k in range(4):
#             begin_index = k * sub_len
#             end_index = (k+1) * sub_len
#
#             if begin_index > len(sample):
#                 continue
#
#             sub_sample = sample[begin_index: end_index]
#             split_inputs.append(sub_sample)
#             if before_sub_sample is not None:
#                 before_contexts.append({'id': idx,
#                                         'input': tokenizer.decode(before_sub_sample)})
#
#             before_sub_sample = sub_sample
#             idx += 1
#
#         if len(split_inputs) % 10000 == 0:
#             logger.info(f"{len(split_inputs)} samples")
#
#     return split_inputs, before_contexts


class RetrieveDataset(Dataset):
    def __init__(self, tokenizer, lang, file_path, block_size=512, api=True):
        self.tokenizer = tokenizer
        #self.args = args
        self.api = api
        self.block_size = block_size
        data_file = file_path

        if lang == "java":
            from process_java import processor
        elif lang == "python":
            from process_python import processor
        self.proc = processor(lang, remove_comments=False)

        logger.info(f"Creating features from {data_file}")
        data_format = data_file.split(".")[-1]

        self.data = []
        self.idx = []
        n = 0
        with open(data_file) as f:
            for _ in f:
                n += 1
        # n = 100000
        st = 0
        ed = n
        logger.warning(f"device -1 will load {ed - st} data line from {st} to {ed}")
        with open(data_file) as f:
            for i, line in enumerate(f):
                if i >= st and i < ed:
                    if (i - st) % 100000 == 0:
                        logger.info(f"device -1 created {i - st}/{ed - st} train data")
                    if "json" in data_format:
                        content = json.loads(line)
                        self.data.append(self.convert_cxg_format_to_normal(content["input"]))
                        self.idx.append(content["id"])
                    else:  # txt
                        self.data.append(self.convert_cxg_format_to_normal(line.strip()))
                        self.idx.append(i)
        logger.warning(f"device -1 loaded {len(self.data)} train data from {st} to {ed}")

    def convert_cxg_format_to_normal(self, code):
        if code.startswith("<s>"):
            code = code.lstrip("<s>")
        if code.endswith("</s>"):
            code = code.rstrip("</s>")
        code = code.replace("<EOL>", "\n")
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        return code

    def encode(self, code, api_seq):
        if self.api:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + \
                          [self.tokenizer.sep_token] + self.tokenizer.tokenize(" ".join(api_seq)) + [
                              self.tokenizer.sep_token]
        else:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + [self.tokenizer.sep_token]
        code_tokens = code_tokens[:self.block_size]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return code_ids

    def process(self, code):
        self.proc.update(code)
        api_seq = self.proc.get_api_seq()
        code = self.proc.untokenize(cut_ratio=0.0)
        token_id = self.encode(code, api_seq)
        return token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.process(self.data[item])), torch.tensor([self.idx[item]])


class TokenCompletionDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024,
                 load_file=None, search_res=None, before_code_file=None):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.file_type = file_type
        self.tokenizer = tokenizer

        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size))

        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                data = pickle.load(handle)
        else:
            logger.info("please separate data first.")
            raise EOFError

        length = len(data)
        logger.info("Data size: %d" % (length))

        self.inputs = []
        self.token_labels = []

        before_contexts = {}
        if before_code_file is not None:
            with open(args.dstore_path + "/" + before_code_file, 'r') as f:
                for line in f.readlines():
                    line = json.loads(line)
                    before_contexts[line['id']] = line['input']
        else:
            raise EOFError

        # load_file = train_split
        # search_res = 保存的索引结果
        # nexts = 存储数据库中下一个代码片段索引
        dstore_path = args.dstore_path
        if load_file is not None:
            id2code = {}  # 存储数据库代码索引
            lines = open(os.path.join(dstore_path, load_file+".txt")).readlines()
            for i,line in enumerate(tqdm(lines)):
                id2code[i] = line.strip()

            search_results = pickle.load(open(os.path.join(dstore_path, search_res + ".pkl"), "rb"))
            try:
                nexts = pickle.load(open(os.path.join(dstore_path, load_file+"_nexts.pkl"), "rb"))
            except Exception:
                nexts = [i for i in range(len(lines))]

        for i, token_ids in enumerate(tqdm(data)):
            if i % 1000 == 0:
                logger.info(f"Encoded {i}/{length} data")

            if i in before_contexts:
                before_context = before_contexts[i]
                before_tokens = [t for t in tokenizer.tokenize(before_context)]
                before_code_ids = tokenizer.convert_tokens_to_ids(before_tokens)
            else:
                before_code_ids = []

            if load_file is not None:
                try:
                    if i in search_results:
                        cand_id = search_results[i]
                    else:
                        cand_id = search_results[str(i)]
                    cand = id2code[cand_id]
                    if nexts[cand_id] != cand_id:
                        cand += id2code[nexts[cand_id]]
                    #cand = tokenizer.encode(cand)
                    cand_tokens = [t for t in tokenizer.tokenize(cand)]
                    cand = tokenizer.convert_tokens_to_ids(cand_tokens)
                except:
                    # print("OK") 没有前缀
                    cand = []
            else:
                cand = []

            # print(tokenizer.decode(before_code_ids))
            # print(tokenizer.decode(cand))
            input_ids = cand + before_code_ids + token_ids
            token_labels = [1] * (len(cand) + len(before_code_ids)) + [2] * len(token_ids)

            input_ids = input_ids[-block_size:]
            token_labels = token_labels[-block_size:]

            self.inputs.append(input_ids)
            self.token_labels.append(token_labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


class LineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924, load_file=None, search_res=None):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        if load_file is not None:
            id2code = {}
            lines = open(os.path.join(args.dstore_path, load_file+".txt")).readlines()
            for i,line in enumerate(tqdm(lines)):
                id2code[i] = line.strip()

            search_results = pickle.load(open(os.path.join(args.dstore_path, search_res + ".pkl"), "rb"))
            try:
                nexts = pickle.load(open(os.path.join(args.dstore_path, load_file+"_nexts.pkl"), "rb"))
            except Exception:
                nexts = [i for i in range(len(lines))]

        length = len(datas)
        logger.info("Data size: %d"%(length))
        self.inputs = []
        self.gts = []
        for i,data in enumerate(datas):
            if i % 1000 == 0:
                logger.info(f"Encoded {i}/{length} data")
            data = json.loads(data.strip())
            if load_file is not None:
                try:
                    data_id = data["id"] if "id" in data else str(i)
                    cand_id = search_results[data_id]
                    cand = id2code[cand_id]
                    if nexts[cand_id] != cand_id:
                        cand += id2code[nexts[cand_id]]
                    #cand = tokenizer.encode(cand)
                    cand_tokens = [t for t in tokenizer.tokenize(cand)]
                    cand = tokenizer.convert_tokens_to_ids(cand_tokens)
                except:
                    cand = []
            else:
                cand = []
            input_tokens = [t for t in tokenizer.tokenize(data["input"])]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            self.inputs.append((cand + input_ids)[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]
