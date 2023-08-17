import argparse
import csv
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from dataset import RetrieveDataset, build_token_completion_data, TokenCompletionDataset
import logging
import os
import json
import numpy as np
import faiss

import time

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

from model import UnixCoderLM

from transformers import (
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)

logger = logging.getLogger(__name__)


# return MB size
import os
from os.path import join, getsize

def get_dir_size(dir_path):
    dir_size = 0
    for root, dirs, files in os.walk(dir_path):
        dir_size += sum([getsize(join(root, name)) for name in files])
    return round(dir_size / 1024 / 1024, 2)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'unixCoder': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
}


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


# 将需要存储到数据库的数据进行分块
def split_code(file_name, output_dir, max_chunk_len=300):
    print('split code from ', file_name)
    lines = open(file_name, "r").readlines()
    split_file_path = output_dir + '/' + file_name.split("/")[-1].split(".")[0] + "_split.txt"
    wf = open(split_file_path, "w")
    nexts = []
    cnt = 0
    for line in lines:
        #tokens = line.strip().split()
        # intra_project 是json格式存储
        tokens = json.loads(line.strip())
        s = len([t for t in tokens if '<STR_LIT' in t])

        if s > 1024:
            continue

        if len(tokens) <= max_chunk_len:
            wf.write(" ".join(tokens) + "\n")
            nexts.append(cnt)
            cnt += 1
        else:
            for i in range(0, len(tokens), max_chunk_len):
                wf.write(" ".join(tokens[i:i + max_chunk_len]) + "\n")
                nexts.append(cnt + 1)
                cnt += 1
            nexts[-1] -= 1
    wf.close()
    pickle.dump(nexts, open(output_dir + '/' + file_name.split("/")[-1].split(".")[0] + "_split_nexts.pkl", "wb"))

    return split_file_path


def my_collect_fn(sequences, batch_first=True, padding_value=1):
    inputs = []
    inputs1 = []
    for (x, x1) in sequences:
        inputs.append(x)
        inputs1.append(x1)
    return (
        pad_sequence(inputs, batch_first, padding_value),
        pad_sequence(inputs1, batch_first, padding_value),
    )


def save_vec(args, file, tokenizer, model, save_name, output_path, lang='python', api=True):
    # build dataloader
    print('save vectors from ', file)
    eval_batch_size = 8
    dataset = RetrieveDataset(tokenizer, lang,
                              file, block_size=512, api=api)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size,
                            collate_fn=partial(my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id),
                            num_workers=4)

    model.to(args.device)

    # Eval!
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()

    steps = 0
    # num_vec = 8
    n_vec = 0
    saved = {}
    logger.info(f"get vectors from {file}")
    for batch in dataloader:
        with torch.no_grad():
            (inputs1, inputs2) = batch
            inputs1 = inputs1.to(args.device)
            attn_mask1 = torch.tensor(inputs1.clone().detach() != tokenizer.pad_token_id, dtype=torch.uint8,
                                      device=args.device)
            outputs = model(inputs1, attention_mask=attn_mask1)[0]
            if n_vec > 0:
                outputs = nn.functional.normalize(outputs[:, :n_vec, :], dim=2)
            else:
                outputs = nn.functional.normalize(outputs[:, 0, :], dim=1)
            outputs = outputs.detach().to("cpu").numpy()
            idxs = inputs2.numpy()
        for i in range(outputs.shape[0]):
            saved[idxs[i][0]] = outputs[i]
        steps += 1
        if steps % 100 == 0:
            logger.info(f"Inferenced {steps} steps")

    file_path = output_path + '/' + save_name + ".pkl"

    pickle.dump(saved, open(file_path, "wb"))

    return file_path


def search_bm25(corpus_file, query_file, temp_dir, db_name, save_path):
    print("Building bm25 corpus")

    datas = open(corpus_file).readlines()
    lines = open(query_file).readlines()
    try:
        os.mkdir(temp_dir)
    except FileExistsError:
        pass
    fidx = open(os.path.join(temp_dir, "corpus.jsonl"), "w")
    #fq = open(os.path.join(temp_dir, "query.jsonl"), "w")
    #fr = open(os.path.join(temp_dir, "res.tsv"), "w")

    for i, line in enumerate(tqdm(datas)):
        fidx.write(json.dumps({"_id": str(i), "text": line.strip()}) + "\n")
    with open(os.path.join(temp_dir, "res.tsv"), "w") as fr, open(os.path.join(temp_dir, "query.jsonl"), "w") as fq:
        csv_fr = csv.writer(fr, delimiter='\t')
        fr.write("q\td\t\s\n")
        for i, line in enumerate(tqdm(lines)):
            content = json.loads(line)
            idx = content["id"] if "id" in content else str(i)
            csv_fr.writerow([str(idx), str(idx), 1])
            code = content["input"].strip()
            fq.write(json.dumps({"_id": str(idx), "text": code}) + "\n")
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=os.path.join(temp_dir, "corpus.jsonl"),
            query_file=os.path.join(temp_dir, "query.jsonl"),
            qrels_file=os.path.join(temp_dir, "res.tsv")
        ).load_custom()

        model = BM25(index_name=db_name, hostname="localhost:9200", initialize=True)
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)
        pickle.dump(results, open(save_path, "wb"))
    except Exception as e:
        print('出错了！！！！')
        print(e)


def search_dense(index_file, query_file, save_name):
    index_data = pickle.load(open(index_file, "rb"))
    query_data = pickle.load(open(query_file, "rb"))
    ids = []
    indexs = []
    id2n = {}
    for i, (idx, vec) in enumerate(index_data.items()):
        ids.append(idx)
        indexs.append(vec)
        id2n[idx] = i
    queries = []
    idxq = []
    for idx, vec in query_data.items():
        queries.append(vec)
        idxq.append(idx)
    ids = np.array(ids)
    indexs = np.array(indexs)
    print('index shape is ', indexs.shape)

    queries = np.array(queries)

    # build faiss index
    d = 768
    k = 101
    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    index_id = faiss.IndexIDMap(index)
    index_id.add_with_ids(indexs, ids)

    res = {}
    D, I = index_id.search(queries, k)
    for i, (sd, si) in enumerate(zip(D, I)):
        res[str(idxq[i])] = {}
        for pd, pi in zip(sd, si):
            res[str(idxq[i])][str(pi)] = pd

    pickle.dump(res, open(save_name, "wb"))


def hybrid_scores(bm25_scores, dense_scores, alpha, beilv=100):
    # beilv: re-scaling dense score as it is percentage.
    scores = {}
    for idx, v in tqdm(dense_scores.items()):
        new_v = {}
        if idx not in bm25_scores:
            scores[idx] = v
            continue
        v2 = bm25_scores[idx]
        v_min = min(list(v.values()))
        v2_min = min(list(v2.values()))
        for _id, score in v.items():
            if _id not in v2:
                new_v[_id] = beilv * score + alpha * v2_min
            else:
                new_v[_id] = beilv * score + alpha * v2[_id]
        for _id, score in v2.items():
            if _id not in new_v:
                new_v[_id] = alpha * score + beilv * v_min
        scores[idx] = new_v
    return scores


def get_res(bm25_file, dense_file, save_file, alpha):
    if bm25_file != "":
        bm25_scores = pickle.load(open(bm25_file, "rb"))
        print("bm25 scores loaded")
    else:
        bm25_scores = {}
    if dense_file != "":
        dense_scores = pickle.load(open(dense_file, "rb"))
        print("dense scores loaded")
    else:
        dense_scores = {}

    res = {}
    if len(bm25_scores) > 0 and len(dense_scores) > 0:
        scores = hybrid_scores(bm25_scores, dense_scores, alpha, 100)
    elif len(bm25_scores) > 0:
        scores = bm25_scores
    else:
        scores = dense_scores
    for idx, v in tqdm(scores.items()):
        v = sorted(v.items(), key=lambda x: -x[1])
        # res[int(idx)] = int(v[0][0]) if v[0][0] != idx else int(v[1][0])
        res[int(idx)] = int(v[0][0])

    pickle.dump(res, open(save_file, "wb"))


def load_pretrained_model(args, special_tokens):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # logger.info("model type: ", args.model_type)
    # print(args.pretrain_dir)
    # test = RobertaTokenizer.from_pretrained(args.pretrain_dir)
    # print('load test.')
    tokenizer = tokenizer_class.from_pretrained(args.pretrain_dir, do_lower_case=False, sep_token='<EOL>',
                                                bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
    if args.model_type == "unixCoder":
        config = config_class.from_pretrained("microsoft/unixcoder-base")
        config.is_decoder = True
        decoder = model_class.from_pretrained("microsoft/unixcoder-base",
                                              config=config)
        decoder.resize_token_embeddings(len(tokenizer))

        model = UnixCoderLM(decoder, config, pad_id=tokenizer.pad_token_id)

        model_last = os.path.join(args.pretrain_dir, 'model.pt')
        if os.path.exists(model_last):
            logger.warning(f"Loading model from {model_last}")
            model.load_state_dict(torch.load(model_last, map_location="cpu"))
    else:
        model = model_class.from_pretrained(args.pretrain_dir)
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def load_retriever():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/reacc-py-retriever")
    model = RobertaModel.from_pretrained("microsoft/reacc-py-retriever", add_pooling_layer=False)
    return tokenizer, model