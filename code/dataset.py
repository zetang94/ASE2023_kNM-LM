# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import os
import pickle
import gc
import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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
    #source_ids = tokenizer.encode(source, add_special_tokens=False)
    return source_ids


class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size) + "_wordsize_%d" % (
            world_size) + "_rank_%d" % (local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d" % (length))
            input_ids = []
            for idx, x in enumerate(data):
                try:
                    x = json.loads(x)
                    s = len([t for t in x if '<STR_LIT' in t])

                    if s > 1024:
                        continue

                    x = ' '.join(x)
                    ids = tokenize(x, tokenizer, args.model_type)
                    input_ids.extend(ids)
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("Rank %d, load %d" % (local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids) // world_size
            logger.info(f"tokens: {length * world_size}")
            input_ids = input_ids[local_rank * length: (local_rank + 1) * length]

            for i in range(0, length - block_size, block_size):
                self.inputs.append(input_ids[i: i + block_size])
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples" % (local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class TokenCompletionDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.file_type = file_type

        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d" % (length))
            input_ids = []
            for idx, x in enumerate(data):
                try:
                    x = json.loads(x)
                    s = len([t for t in x if '<STR_LIT' in t])

                    if s > 1024:
                        continue

                    x = ' '.join(x)

                    ids = tokenize(x, tokenizer, args.model_type)

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
                                if self.file_type == 'train':
                                    i = i + block_size
                                    continue
                                print(tokenizer.decode(sample))
                                exit()
                            sample = sample[: block_size - 1 - j]

                            i += len(sample)
                            pad_len = block_size - len(sample)
                            sample += [tokenizer.pad_token_id] * pad_len
                            # sample = bos_ids + sample

                            input_ids.append(sample)
                        else:
                            pad_len = block_size - len(sample)
                            sample += [tokenizer.pad_token_id] * pad_len
                            # sample = bos_ids + sample
                            input_ids.append(sample)
                            break
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            #logger.info(f"tokens: {len(input_ids)}")
            #self.split(input_ids, tokenizer, logger, model_type=args.model_type, block_size=block_size)
            self.inputs = input_ids
            #del input_ids
            #gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class LineCompletionDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")

        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            with open(datafile) as f:
                datas = f.readlines()

            length = len(datas)
            logger.info("Data size: %d"%(length))
            self.inputs = {'sources': [], 'gts': []}
            for data in tqdm(datas):
                data = json.loads(data)
                source = data['input']
                token_ids = tokenize(source, tokenizer, args.model_type)[-block_size:]
                self.inputs['sources'].append(token_ids)
                self.inputs['gts'].append(data["gt"])

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs['gts'])

    def __getitem__(self, item):
        return torch.tensor(self.inputs['sources'][item]), self.inputs['gts'][item]