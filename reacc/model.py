# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)
        self.criterion = nn.CrossEntropyLoss()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden=None, labels=None):
        emb = self.encoder(input)
        if hidden is not None:
            output, hidden = self.rnn(emb, hidden)
        else:
            output, hidden = self.rnn(emb)
        output = self.drop(output)
        output = self.decoder(output)
        # decoded = decoded.view(-1, self.ntoken)
        # output = F.log_softmax(decoded, dim=1)
        if labels is not None:
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, output, hidden
        else:
            return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class UnixCoderLM(nn.Module):
    def __init__(self, decoder, config, pad_id):
        super(UnixCoderLM, self).__init__()
        self.decoder = decoder
        self.config = config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embeddings.word_embeddings.weight
        #self.lsm = nn.Softmax(dim=-1)
        self.pad_id = pad_id

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=None, past_key_values=None):
        length = input_ids.size(-1)
        transformer_outputs = self.decoder(input_ids, attention_mask=self.bias[:, :length, :length],
                                           past_key_values=past_key_values)
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        past_key_values = transformer_outputs.past_key_values

        # if isinstance(outputs, Tuple):
        #     lm_logits, _lambda = outputs
        #     output = (lm_logits, past_key_values, _lambda)   # 这个是在knn中做完softmax再输出
        # else:
        #     lm_logits = outputs
        output = (lm_logits, past_key_values)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            active_loss = input_ids[..., 1:].ne(1).view(-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

        return ((loss,) + output) if loss is not None else output

    