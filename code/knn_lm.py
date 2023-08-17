import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
import math
import gc
import faiss
import faiss.contrib.torch_utils
import ctypes
logger = logging.getLogger(__name__)
logger.setLevel(20)


class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper(object):
    def __init__(self, dstore_size, dstore_dir, dimension,
                 knn_sim_func=None, knn_keytype=None,
                 no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
                 recompute_dists=False,
                 k=1024, lmbda=0.25, knn_temp=1.0, probe=32, use_bayes=False,
                 window_size=32, pad_id=None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.cache_total = None
        self.hook_handles = []

        self.use_bayes = use_bayes
        self.window_size = window_size
        self.pad_id = pad_id

        # 为了行级别补全调整
        self.original_use_bayes = self.use_bayes
        self.original_lmbda = self.lmbda
        self.cur_lambda = None

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func]  # l2 or dot product function

    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        if isinstance(cpu_index, faiss.IndexIVFPQ):
            cpu_index.nprobe = self.probe

        if self.knn_gpu:
            logger.info('Use GPU for knn.')
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n),
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        if isinstance(cpu_index, faiss.IndexIVFPQ):
            cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size,
                                           self.dimension)
        if not self.no_load_keys:
            self.keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy',
                                                  dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                                         shape=(self.dstore_size, 1))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][
            self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        self.input_ids = input_ids
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    # 行级别补全需要更新
    def update_param(self, use_bayes):
        self.use_bayes = use_bayes
        if isinstance(self.cur_lambda, float):
            _lmbda = self.cur_lambda
        else:
            _lmbda = self.cur_lambda.squeeze(-1)[-1].item()   # 只支持batch_size=1!!!, 为行级别补全调整
        self.lmbda = _lmbda

    def reset(self):
        self.use_bayes = self.original_use_bayes
        self.lmbda = self.original_lmbda

    # 行级别补全更新后需要还原

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.softmax(lm_logits, dim=-1).flatten(0, 1)
        queries = self.activation_capturer.captured.flatten(0, 1)  # (batch, time, dim)

        dists, knns = self.get_knns(queries)

        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(queries, knns_vecs)

        # (batch*time, dim)
        knn_log_probs, _ = self.knns_to_prob(knns, dists)

        if self.use_bayes:
            p_knn_index = torch.argmax(knn_log_probs, dim=-1).view(batch, -1)
            p_lm_index = torch.argmax(lm_logits, dim=-1).view(batch, -1)

            # 只看前面的概率,最后一个位置的输出没有用，所以不用管
            before_believe_knn = (p_knn_index[:, :-shift] == self.input_ids[:, shift:])
            before_believe_lm = (p_lm_index[:, :-shift] == self.input_ids[:, shift:])
            believe_knn = before_believe_knn * ~before_believe_lm
            believe_lm = before_believe_lm * ~before_believe_knn

            # window_size 8
            believe_lm = self.n_gram(believe_lm, n=int(np.log2(self.window_size)))
            believe_knn = self.n_gram(believe_knn, n=int(np.log2(self.window_size)))

            zeros = torch.ones(batch, 1).to(self.device)
            believe_knn = torch.cat([zeros, believe_knn], dim=1)
            believe_lm = torch.cat([zeros, believe_lm], dim=1)

            error_rate = self.lmbda * self.window_size

            _lambda = (believe_knn + error_rate) / (believe_knn + believe_lm + self.window_size)
            _lambda = _lambda.contiguous().view(-1).unsqueeze(-1)
        else:
            _lambda = self.lmbda

        self.cur_lambda = _lambda

        output = (1-_lambda) * lm_logits + _lambda * knn_log_probs
        output = output.view(batch, time_dim, -1)
        return output

    # def post_forward_hook(self, module, input, output):
    #     batch, time_dim, vocab_size = output.shape
    #     shift = 0 if self.is_encoder_decoder else 1
    #     lm_logits = output
    #     #lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1)  # (batch, time, vocab)
    #     lm_logits = torch.softmax(lm_logits, dim=-1)
    #     queries = self.activation_capturer.captured  # (batch, time, dim)
    #     if self.labels is None:
    #         nonpad_mask = torch.cat([
    #             torch.ones([batch, time_dim - 1], dtype=torch.bool),
    #             torch.zeros([batch, 1], dtype=torch.bool),
    #         ], axis=-1).to(self.device)
    #     else:
    #         nonpad_mask = torch.cat([
    #             self.labels[:, shift:] != self.pad_id,
    #             torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
    #         ], axis=-1)
    #
    #     lm_logits = lm_logits[..., :-1, :]
    #
    #     lm_logits = lm_logits[nonpad_mask]
    #     queries = queries[nonpad_mask]  # (nonpad, dim)
    #
    #     dists, knns = self.get_knns(queries)  # (nonpad batch * time, k)
    #     if self.recompute_dists:
    #         knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
    #         dists = self.dist_func(queries, knns_vecs)
    #
    #     knn_log_probs, _ = self.knns_to_prob(knns, dists)
    #
    #     if self.use_bayes:
    #         p_knn_index = torch.argmax(knn_log_probs, dim=-1).view(batch, -1)
    #         p_lm_index = torch.argmax(lm_logits, dim=-1).view(batch, -1)
    #
    #         # 只看前面的概率,最后一个位置的输出没有用，所以不用管
    #         before_believe_knn = (p_knn_index == self.input_ids[:, shift:])
    #         before_believe_lm = (p_lm_index == self.input_ids[:, shift:])
    #         believe_knn = before_believe_knn * ~before_believe_lm
    #         believe_lm = before_believe_lm * ~before_believe_knn
    #
    #         # window_size 8
    #         believe_lm = self.n_gram(believe_lm, n=int(np.log2(self.window_size)))
    #         believe_knn = self.n_gram(believe_knn, n=int(np.log2(self.window_size)))
    #
    #         zeros = torch.ones(batch, 1).to(self.device)
    #         believe_knn = torch.cat([zeros, believe_knn], dim=1)
    #         believe_lm = torch.cat([zeros, believe_lm], dim=1)
    #
    #         error_rate = self.lmbda * self.window_size
    #
    #         _lambda = (believe_knn + error_rate) / (believe_knn + believe_lm + self.window_size)
    #         _lambda = _lambda[:, :-1].contiguous().view(-1).unsqueeze(-1)
    #     else:
    #         _lambda = self.lmbda
    #
    #     interpolated_scores = (1-_lambda) * lm_logits + _lambda * knn_log_probs
    #
    #     output[nonpad_mask] = interpolated_scores
    #     return output

    def knns_to_prob(self, knns, dists):
        p_knn = torch.zeros(knns.size(0), self.vocab_size).to(knns.device)
        logits = torch.softmax(-1 * torch.pow(dists, 0.5) / 3, dim=-1)
        size = knns.size()
        neighbour_targets = self.vals[knns].squeeze(-1)
        neighbour_targets = neighbour_targets.view(size)

        p_knn = torch.scatter_add(p_knn, 1, neighbour_targets, logits)

        return p_knn, neighbour_targets


    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    @staticmethod
    def n_gram(logits, n=3):
        """n gram, n = 2^n"""
        for i in range(n):
            z = np.power(2, i)
            logits[:, z:] = logits[:, :-z] + logits[:, z:]  # 2^(z-1)~2^z-1
        # logits = logits + 1
        return logits

    def get_metrics(self):
        return {}

    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        if type(lmbda) is float:
            log = math.log
        else:
            log = torch.log
        interpolated = torch.logaddexp(
            lm_log_probs + log(1 - lmbda),
            knn_log_probs + log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer,
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model,
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'roberta': {
            KEY_TYPE.last_ffn_input: (lambda model: model.lm_head, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.lm_head, False),
        },

        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        }
    }


class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension,
                 knn_keytype=None, pad_id=None, need_knn_train=False, only_errors=False):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None

        self.dstore_keys = None
        self.key_file_name = None
        self.dstore_vals = None
        self.dstore_idx = 0
        self.only_errors = only_errors

        self.labels = None
        self.pad_id = pad_id
        self.hook_handles = []
        self.need_knn_train = need_knn_train

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][
            self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size,
                                           self.dimension)

        keys_file_name = f'{keys_vals_prefix}_keys.npy'
        self.key_file_name = keys_file_name

        vals_filename = f'{keys_vals_prefix}_vals.npy'

        if os.path.exists(keys_file_name) and os.path.exists(vals_filename):
            logger.error("keys already exist:" + keys_file_name)
            mode = 'r'
        else:
            mode = 'w+'

        self.dstore_keys = np.memmap(keys_file_name, dtype=np.float16, mode=mode,
                                     shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode,
                                     shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError(
                'labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured

        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1)  # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1)  # (batch * time)

        nonpad_mask = captured_values != self.pad_id

        lm_logits = output
        pred_ids = torch.argmax(lm_logits[..., :-shift, :], dim=-1)  # (batch, time, vocab)
        pred_ids = pred_ids.flatten(0, 1)

        if self.only_errors:
            nonpad_mask = (nonpad_mask * (pred_ids != captured_values)).bool()

        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        token_num = keys.shape[0]

        if self.dstore_idx + token_num > self.dstore_size:
            token_num = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:token_num]
            values = values[:token_num]

        try:
            self.dstore_keys[self.dstore_idx: (token_num + self.dstore_idx)] = keys.cpu().numpy().astype(
                np.float16)
            self.dstore_vals[self.dstore_idx:(token_num + self.dstore_idx)] = values.unsqueeze(
                -1).cpu().numpy().astype(np.int32)

        except ValueError as ex:
            logger.error(
                f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex

        self.dstore_idx += token_num
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000,
                    ncentroids=4096, seed=1, code_size=64, probe=32):
        # to speed up access to np.memmap

        tokens_index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        actual_tokens_size = read_or_load_datastore_size(self.dstore_dir, 'r')

        np.random.seed(seed)

        if self.need_knn_train:
            logger.info("Building tokens index.")
            build_datastore(self.dimension, ncentroids, code_size, probe, True, tokens_index_name,
                            self.key_file_name, actual_tokens_size,
                            num_keys_to_add_at_a_time=num_keys_to_add_at_a_time)
        else:
            logger.info("Building tokens index.")
            self.add_index(self.key_file_name, actual_tokens_size, tokens_index_name)

        del self.dstore_keys
        gc.collect()
        os.remove(self.key_file_name)

    def add_index(self, key_path, actual_size, index_name):
        if not os.path.exists(key_path):
            logger.error("key file not exists")
            return
        keys = np.memmap(key_path, dtype=np.float16, mode='r',
                         shape=(actual_size, self.dimension))
        index = faiss.IndexFlatL2(self.dimension)
        logger.info('Adding Keys')
        #ids = torch.arange(actual_size)
        index.add(torch.tensor(keys[:actual_size].astype(np.float32)))
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')

    def write_flush(self):
        self.dstore_keys.flush()
        self.dstore_vals.flush()
        read_or_load_datastore_size(self.dstore_dir, 'w', self.dstore_idx)
        self.dstore_keys = None
        gc.collect()

def read_or_load_datastore_size(dstore_dir, mode=None, data_size=None):
    file_name = os.path.join(dstore_dir, 'data_store_size.txt')
    if mode == 'w' and data_size is not None:
        with open(file_name, 'w') as f:
            f.write(str(data_size))
            return data_size
    else:
        with open(file_name, 'r') as f:
            data_size = f.readline()
            return int(data_size)

def get_metrics(self):
        return {}


def build_datastore(index_dim, ncentroids, code_size, probe, use_gpu, faiss_index,
                    key_path, dstore_size,
                    num_keys_to_add_at_a_time=1000000):
    # faiss参数: index_dim, ncentroids, code_size, probe
    # faiss_index: 存储train_index名称
    # random_samples: 训练数据
    # keys: 需要存储的数据 dstore_size : 存储数据大小
    if not os.path.exists(key_path):
        logger.error("key file not exists")
        return
    keys = np.memmap(key_path, dtype=np.float16, mode='r',
                                     shape=(dstore_size, index_dim))

    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize,
                   1) == 0, "MADVISE FAILED"  # 2 means MADV_SEQUENTIAL

    random_error_ids = np.random.choice(np.arange(keys.shape[0]),
                                        size=[min(1000000, dstore_size)], replace=False)

    random_samples = keys[random_error_ids].astype(np.float32)

    res = faiss.StandardGpuResources()

    if not os.path.exists(faiss_index + ".trained"):
        quantizer = faiss.IndexFlatL2(index_dim)
        index = faiss.IndexIVFPQ(quantizer, index_dim, ncentroids, code_size, 8)
        index.nprobe = probe

        if use_gpu:
            logger.info('Start put index to gpu')
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

        logger.info('Training Index')
        start = time.time()

        if use_gpu:
            gpu_index.train(random_samples.astype(np.float32))
        else:
            index.train(random_samples.astype(np.float32))

        logger.info('Training took {} s'.format(time.time() - start))

        logger.info('Writing index after training')

        start = time.time()

        if use_gpu:
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_index + ".trained")
        else:
            faiss.write_index(index, faiss_index + ".trained")

        logger.info('Writing index took {} s'.format(time.time() - start))

    logger.info('Adding Keys')
    index = faiss.read_index(faiss_index + ".trained")

    if use_gpu:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

    start = 0
    start_time = time.time()
    while start < dstore_size:
        end = min(dstore_size, start + num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()

        if use_gpu:
            gpu_index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
        else:
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))

        start += num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            print('Added %d tokens so far' % start)
            print('Writing Index', start)
            if use_gpu:
                faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_index)
            else:
                faiss.write_index(index, faiss_index)

    logger.info("Adding total %d keys" % end)
    logger.info('Adding took {} s'.format(time.time() - start_time))
    logger.info('Writing Index')
    start = time.time()

    if use_gpu:
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_index)
    else:
        faiss.write_index(index, faiss_index)

    logger.info('Writing index took {} s'.format(time.time() - start))


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'


def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'

