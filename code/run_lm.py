from __future__ import absolute_import, division, print_function

import multiprocessing
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import time
import pandas
from clearml import Task, Logger
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import get_dir_size
from model import UnixCoderLM
from config import add_args
from dataset import FinetuneDataset, LineCompletionDataset, TokenCompletionDataset
from knn_lm import DIST, KEY_TYPE, KNNWrapper, KNNSaver
from beam import Beam
import logging
from fuzzywuzzy import fuzz

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'unixCoder': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
}


cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size


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


def train(args, train_dataset, model, tokenizer, fh, pool):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    total_examples = len(train_dataset) * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # batch_size = args.batch_size * args.gradient_accumulation_steps * (
    #     torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        logger.warning(f"Loading optimizer from {optimizer_last}")
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples)
    logger.info("  Num epoch = %d", t_total * batch_size // total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0, global_step
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
            #if (step + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.logging_steps == 0:
                    logger.info("  steps: %s  ppl: %s  lr: %s", global_step, round(avg_loss, 5),
                                scheduler.get_last_lr()[0])
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate_sub_word(args, model, tokenizer, eval_when_training=True, file_type='dev', need_acc=False)

                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,
                                                                                     round(results['perplexity'], 4)))
                    else:
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    # if args.model_type == "rnn" or args.model_type == 'unixCoder':
                    #     torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                    # else:
                    #     model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    #
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    Logger.current_logger().report_scalar(
                        "Training loss", "model", iteration=step, value=results['perplexity']
                    )
                    # _rotate_checkpoints(args, checkpoint_prefix)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    if args.model_type == "rnn" or args.model_type == 'unixCoder':
                        torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "model.pt"))
                    else:
                        model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate_sub_word(args, model, tokenizer, prefix="", eval_when_training=False, file_type='dev', need_acc=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)    eval_output_dir = args.output_dir

    eval_dataset = TokenCompletionDataset(tokenizer, args, logger, file_type=file_type,
                                          block_size=args.block_size)

    # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) # if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

    model.to(args.device)
    # multi-gpu evaluate
    # if args.n_gpu > 1 and eval_when_training is False:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    total_acc = 0
    total_token_num = 0
    model.eval()

    for batch in tqdm(eval_dataloader, f'Evaluating on {file_type} dataset.'):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()

            # token acc
            if need_acc:
                shift = 1
                pred_scores = outputs[1]
                pred_ids = pred_scores.argmax(-1)
                pred_ids = pred_ids[:, :-shift].flatten(0, 1)
                labels = labels[:, shift:].flatten(0, 1)
                mask_pad = labels != tokenizer.pad_token_id
                batch_num = torch.sum(mask_pad)
                batch_acc = torch.sum(pred_ids[mask_pad] == labels[mask_pad])
                total_acc += batch_acc
                total_token_num += batch_num

                if nb_eval_steps % 100 == 1:
                    print(f'{total_token_num} tokens, acc {total_acc / total_token_num}')

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))


    result = {
        "perplexity": float(perplexity)
    }
    if need_acc:
        acc = total_acc / total_token_num
        result["acc"]=float(acc)

    return result


def evaluate_word_acc(args, model, tokenizer, file_type='test'):
    eval_dataset = TokenCompletionDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node)

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    model.eval()

    correct = 0.0
    total = 0

    total_pred = []
    total_gt = []

    start_time = time.time()
    tmp_pred = []
    tmp_gt = []

    for step, batch in enumerate(eval_dataloader):
        inputs = batch.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
            pred_scores = outputs[0]
            pred_ids = pred_scores.argmax(-1)

        all_pred = []
        all_gt = []
        prev_pred = None
        for pred, gt in zip(pred_ids, inputs):
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()

            tmp_gt.extend(gt)
            tmp_pred.extend(pred)
            if tokenizer.eos_token_id in gt:
                pred = tmp_pred
                gt = tmp_gt
                for i, y in enumerate(gt):
                    if i == 0:
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id]:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                        else:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                    else:
                        if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                            if len(now_gt) > 0:
                                cur_gt = DecodeIds(now_gt).strip().split()
                                try:
                                    cur_pred = DecodeIds(now_pred).strip().split()
                                    if len(cur_gt) <= len(cur_pred):
                                        cur_pred = cur_pred[:len(cur_gt)]
                                    else:
                                        pad_len = len(cur_gt) - len(cur_pred)
                                        cur_pred = cur_pred + ['SPACE'] * pad_len
                                    all_pred.extend(cur_pred)
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.extend(cur_gt)
                                now_gt = []
                                now_pred = []
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id] \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT") \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<STR_LIT") \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<CHAR_LIT"):
                            if len(now_gt) > 0:
                                cur_gt = DecodeIds(now_gt).strip().split()
                                try:
                                    cur_pred = DecodeIds(now_pred).strip().split()
                                    if len(cur_gt) <= len(cur_pred):
                                        cur_pred = cur_pred[:len(cur_gt)]
                                    else:
                                        pad_len = len(cur_gt) - len(cur_pred)
                                        cur_pred = cur_pred + ['SPACE'] * pad_len
                                    all_pred.extend(cur_pred)
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.extend(cur_gt)
                            now_gt = [y]
                            now_pred = [pred[i - 1]]
                            try:
                                all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                            continue
                        now_gt.append(y)
                        now_pred.append(pred[i - 1])
                tmp_pred = []
                tmp_gt = []

        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        for x, y in zip(all_pred, all_gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1

        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
            logger.info(f"{total}, {correct / total}")

    # pickle.dump(total_pred, open(os.path.join(args.output_dir, "preds.pkl"), "wb"))
    # pickle.dump(total_gt, open(os.path.join(args.output_dir, "gts.pkl"), "wb"))

    # saved_file = os.path.join(args.output_dir, "predictions.txt")
    # total_samples = post_process(args, total_pred, total_gt,
    #                              open(os.path.join(args.data_dir, f"{file_type}.txt")).readlines(), saved_file)
    # logger.info(f"Eval on {total_samples}, saved at {saved_file}")
    end_time = time.time()

    prediction_name = args.model_type
    if args.with_knn:
        prediction_name += "__with_knn"
    if args.only_errors:
        prediction_name += "__only_errors"
    if args.use_bayes:
        prediction_name += "__use_bayes"
    prediction_name += "_predictions.txt"

    saved_file = os.path.join(args.output_dir, prediction_name)
    # preds, gts, true_gts, code_types, saved_file
    true_gts, skip_ids = read_true_gts(args.data_dir, file_type)
    total, result = post_process(args, total_pred, total_gt,
                                 true_gts,
                                 read_code_types(args.data_dir, file_type, skip_ids),
                                 saved_file)
    logger.info(f"Eval on {total} tokens, saved at {saved_file}")

    result['time'] = (end_time - start_time)

    return result['avg'], result, total


def read_true_gts(data_dir, file_type):
    true_gts = []
    skip_ids = []
    data = open(os.path.join(data_dir, f"{file_type}.txt")).readlines()
    for id, s in enumerate(data):
        code = json.loads(s)
        tmp = len([t for t in code if '<STR_LIT' in t])
        if tmp > 1024:
            skip_ids.append(id)
            continue
        true_gts.append(code)

    print('true gts', len(true_gts))
    print('skip size', len(skip_ids))
    return true_gts, skip_ids


def read_code_types(data_dir, file_type, skip_ids):
    code_types = []
    data = open(os.path.join(data_dir, f"{file_type}_type.txt")).readlines()
    for id, s in enumerate(data):
        if id in skip_ids:
            continue
        code_type = json.loads(s)
        code_types.append(code_type)
    return code_types


def post_process(args, preds, gts, true_gts, code_types, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []

    total = 0
    correct = 0.0
    code_type_dict = {}
    code_type_correct = {}

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if gt in ["", "<pad>"]:
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if gt == "</s>":
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            true_gt = true_gts[cnt]
            true_gt_str = ' '.join(true_gt).strip()

            if gt_str != true_gt_str:
                with open('t.txt', 'w') as f:
                    f.write(gt_str + '\n')
                    f.write(true_gt_str)

            assert gt_str == true_gt_str, f"{cnt} sample gt_str != true_gt"
            wf.write(pred_str + "\n")

            code_type = code_types[cnt]
            assert len(new_gt) == len(code_type)

            for j, (x, y, z) in enumerate(zip(new_pred, new_gt, code_type)):
                if y not in ["<s>", "</s>", "<EOL>", "<pad>"] and z != 'unknown':
                    total += 1
                    if z not in code_type_dict:
                        code_type_dict[z] = 0
                        code_type_correct[z] = 0
                    code_type_dict[z] += 1
                    if x == y:
                        correct += 1
                        code_type_correct[z] += 1

            cnt += 1
            new_gt = []
            new_pred = []

    code_type_correct = {k: round(v / code_type_dict[k] * 100, 2) for k, v in code_type_correct.items()}
    code_type_dict = {k: round(v / total * 100, 2) for k, v in code_type_dict.items()}

    logger.info(f"Total {total} tokens, accuracy: {round(correct / total * 100, 2)}")
    logger.info(f"Percent code types: " + json.dumps(code_type_dict))
    logger.info(f"Code type accuracy: " + json.dumps(code_type_correct))

    text = args.output_dir + '\n'
    table_name = ''
    table_per = ''
    table_val = ''
    for k in code_type_dict.keys():
        table_name += k + '\t'
        table_per += str(code_type_dict[k]) + '\t'
        table_val += str(code_type_correct[k]) + '\t'
    table_val += str({round(correct / total * 100, 2)}) + '\t'
    text += table_name + '\n' + table_per + '\n' + table_val + '\n'

    result = {}
    for k in code_type_dict.keys():
        result[k + "_percent"] = [code_type_dict[k]]
        result[k] = [code_type_correct[k]]
    result['avg'] = [round(correct / total * 100, 2)]

    file_name = ''
    if args.model_type == "unixCoder":
        if args.only_errors:
            file_name = f'unixCoder_knm_{args.k}_ws_{args.window_size}.txt'
        elif args.with_knn:
            file_name = f'unixCoder_knn_{args.k}_alpha_{args.lmbda}.txt'
        else:
            file_name = 'unixCoder_base_results.txt'
    else:
        if args.only_errors:
            file_name = f'gpt_knm_{args.k}_ws_{args.window_size}.txt'
        elif args.with_knn:
            file_name = f'gpt_knn_{args.k}_alpha_{args.lmbda}.txt'
        else:
            file_name = 'gpt_base_results.txt'

    with open(file_name, 'a+') as f:
        f.write(text)

    return total, result


def eval_line_completion(args, model, tokenizer, file_type='test', knn_wrapper=None):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    dataset = LineCompletionDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size - 100)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    # model.zero_grad()
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'),
                     tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)

            if knn_wrapper is not None:
                knn_wrapper.reset()

            # This is a bug. which means that the last token is fed into the model.
            # model_outputs = model(inputs)
            model_outputs = model(inputs[:, :-1])

            outputs = model_outputs[1]

            if knn_wrapper is not None:
                knn_wrapper.update_param(use_bayes=False)  # 没有观测数据了

            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                if args.model_type == "rnn":
                    past_hidden = tuple(x[:, i:i + 1].expand(-1, beam_size, -1).contiguous() for x in outputs)
                else:
                    past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                            outputs]
                    past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    if args.model_type == "rnn":
                        outputs = model(input_ids, hidden=repackage_hidden(past_hidden))
                    else:
                        outputs = model(input_ids, past_key_values=past_hidden)

                    if knn_wrapper is not None:  # already done softmax
                        out = torch.log(outputs[0][:, -1, :]).data
                    else:
                        out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    if args.model_type == "rnn":
                        past_hidden = tuple(
                            x.data.index_select(1, beam.getCurrentOrigin()).contiguous() for x in outputs[1])
                    else:
                        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for
                                x in outputs[1]]
                        past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                else:
                    text = DecodeIds(t).strip("{").strip()
                #print(text)
                # exit()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
            #logger.info(f"EM: {edit_sim / len(preds)}")
            logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")

    file_name = "prediction_line"
    if args.with_knn and not args.only_errors:
        file_name += "__with_knn"
    if args.only_errors:
        file_name += "__with_knm"
    if args.use_bayes:
        file_name += "__use_bayes"
    file_name += ".txt"
    saved_file = os.path.join(args.output_dir, file_name)
    with open(saved_file, "w") as f:
        for i, (pred_text, gt) in enumerate(zip(preds, gts)):
            if pred_text == gt:
                label = 1
            else:
                label = 0
            save_json = {
                'label': label,
                'pred': pred_text,
                'gt': gt
            }

            f.write(json.dumps(save_json) + "\n")

    logger.info(f"Test {len(preds)} samples")
    logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")

    result = {
        "Edit": float(edit_sim / len(preds)),
        "EM": float(em / len(preds))
    }

    output_eval_file = os.path.join(args.output_dir, "eval_line_result.txt")
    with open(output_eval_file, "w") as writer:
        # logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)

    pool = None
    args = parser.parse_args()

    description = args.pretrain_dir
    clearml_proj_name = args.clearml_proj_name
    # hype_parameter_setting
    if clearml_proj_name.startswith('hype_params'):
        if args.use_bayes:
            description += "__use_bayes"
        if args.with_knn and not args.only_errors:
            description += f"__with_knn_k_{args.k}_lmbda_{args.lmbda}"
        if args.only_errors:
            description += f"__with_knm_k_{args.k}_N_{args.window_size}"

    #    Task.init(project_name="hyparam_unix_android", task_name=args.data_dir + "__" + description)
    else:
        if args.with_knn and not args.only_errors:
            description += "__with_knn"
        if args.only_errors:
            description += "__with_knm"
        if args.use_bayes:
            description += "__use_bayes"

        Task.init(project_name=clearml_proj_name, task_name=args.data_dir + "__" + description)

    # 脚本中先验推测存在问题，需要读取文件acc
    if args.only_errors and not args.build_index:
        path_name = Path(args.dstore_dir)
        acc_dir = str(path_name.parent)
        file = acc_dir+"/acc_in_train_set.txt"
        if os.path.exists(file):
            with open(file, 'r') as f:
                line = f.readline()
                acc = line.split("=")[1].strip()
                acc = float(acc)
                args.lmbda = 1 - acc
                logger.info("[update lmbda] to " + str(args.lmbda))
        else:
            logger.error("No acc in train set file, Stoped.")
            return

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if args.dstore_dir != '' and not os.path.exists(args.dstore_dir):
    #     os.makedirs(args.dstore_dir)

    if args.dstore_dir != '':
        if os.path.exists(args.dstore_dir) and args.build_index:
            shutil.rmtree(args.dstore_dir)
            os.makedirs(args.dstore_dir)
        elif not os.path.exists(args.dstore_dir):
            os.makedirs(args.dstore_dir)




    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.info(
        "local_rank: %d, node_index: %d, gpu_per_node: %d" % (args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')

    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} steps".format(checkpoint_last, args.start_step))

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, sep_token='<EOL>',
                                                    bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                    unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        if args.model_type == "unixCoder":
            config = config_class.from_pretrained("microsoft/unixcoder-base")
            config.is_decoder = True
            decoder = model_class.from_pretrained("microsoft/unixcoder-base",
                                                   config=config)
            decoder.resize_token_embeddings(len(tokenizer))

            model = UnixCoderLM(decoder, config, pad_id=tokenizer.pad_token_id)

            model_last = os.path.join(pretrained, 'model.pt')
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu"))
        else:
            model = model_class.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))

    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>',
                                                    eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                    additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        if args.model_type == "unixCoder":
            model = model_class(len(tokenizer), 768, 768, 1)
        else:
            config = config_class.from_pretrained(args.config_dir)
            model = model_class(config)
            model.resize_token_embeddings(len(tokenizer))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # wrapper knn into model
    dimension = model.config.hidden_size

    knn_saver = None

    if args.build_index:
        knn_saver = KNNSaver(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir,
                             dimension=dimension, knn_keytype=args.knn_keytype,
                             pad_id=tokenizer.pad_token_id, need_knn_train=args.need_knn_train,
                             only_errors=args.only_errors)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Fine-tuning
    if args.do_train:
        train_dataset = FinetuneDataset(tokenizer, args, logger, file_type='train',
                                        block_size=args.block_size)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, fh, pool)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    db_build_time = 0
    db_size = 0

    # build-index
    if args.build_index:
        begin_build_time = time.time()
        if knn_saver is not None:
            knn_saver.break_into(model)

        result = evaluate_sub_word(args, model, tokenizer, eval_when_training=True, file_type='train')
        output_eval_file = os.path.join(args.output_dir, "acc_in_train_set.txt")
        with open(output_eval_file, "w+") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info(' actual dstore size is ' + str(knn_saver.dstore_idx + 1))
        if args.only_errors:
            args.lmbda = 1 - result['acc']
            logger.info("[update lmbda] to " + str(args.lmbda))
        knn_saver.write_flush()
        knn_saver.build_index()
        end_time = time.time()
        knn_saver.break_out()
        del knn_saver

        db_build_time = end_time - begin_build_time

        db_size = get_dir_size(args.dstore_dir)
        #error_db_size = get_dir_size(os.path.join(args.dstore_dir, 'only_errors'))

    if args.do_eval_token:
        if args.with_knn:
            knn_wrapper = KNNWrapper(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir,
                                     dimension=dimension,
                                     knn_sim_func=args.knn_sim_func, knn_keytype=args.knn_keytype,
                                     no_load_keys=args.no_load_keys, move_dstore_to_mem=args.move_dstore_to_mem,
                                     knn_gpu=args.knn_gpu,
                                     recompute_dists=args.recompute_dists,
                                     k=args.k, lmbda=args.lmbda, knn_temp=args.knn_temp, probe=args.probe,
                                     use_bayes=args.use_bayes, window_size=args.window_size,
                                     pad_id=tokenizer.pad_token_id)

            knn_wrapper.break_into(model)

        word_acc, result, total_tokens = evaluate_word_acc(args, model, tokenizer, file_type='test')
        logger.info("word acc: " + str(word_acc))

        result['intra_project'] = [args.data_dir]
        result['description'] = [description]
        result['db_build_time'] = db_build_time
        result['db_size'] = db_size
        #result['all_db_size'] = all_db_size
        result['time'] = total_tokens / result['time']

        # df = pandas.DataFrame(result)
        # df.to_csv(args.model_type + '_projects.csv', mode='a')
        # Logger.current_logger().report_table(
        #     "hyper_parameter",
        #     "PD with index",
        #     table_plot=df
        # )
        results = {}
        file_name = 'hype_result_knm.json' if args.only_errors else 'hype_result_knn.json'
        if os.path.exists(file_name):
            results = json.load(open(file_name,'r'))
        if args.only_errors:
            param = str(args.window_size) + "_" + str(args.k)
        else:
            param = str(args.lmbda) + "_" + str(args.k)

        if param not in results:
            results[param] = []
        results[param].append(word_acc)

        with open(file_name, 'w') as f:
            json.dump(results, f)

    if args.do_eval_line:
        knn_wrapper = None
        if args.with_knn:
            knn_wrapper = KNNWrapper(dstore_size=args.dstore_size, dstore_dir=args.dstore_dir,
                                     dimension=dimension,
                                     knn_sim_func=args.knn_sim_func, knn_keytype=args.knn_keytype,
                                     no_load_keys=args.no_load_keys, move_dstore_to_mem=args.move_dstore_to_mem,
                                     knn_gpu=args.knn_gpu,
                                     recompute_dists=args.recompute_dists,
                                     k=args.k, lmbda=args.lmbda, knn_temp=args.knn_temp, probe=args.probe,
                                     use_bayes=args.use_bayes, window_size=args.window_size,
                                     pad_id=tokenizer.pad_token_id)

            knn_wrapper.break_into(model)

        result = eval_line_completion(args, model, tokenizer, file_type="test", knn_wrapper=knn_wrapper)
        result['intra_project'] = [args.data_dir]
        result['description'] = [description]
        df = pandas.DataFrame(result)
        df.to_csv(args.model_type + '_line_completion_result.csv', mode='a')
        Logger.current_logger().report_table(
            "hyper_parameter",
            "PD with index",
            table_plot=df
        )


if __name__ == '__main__':
    main()













