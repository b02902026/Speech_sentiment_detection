import numpy as np
import torch
from torch import nn
from model import ERBase_


class TERBase_(nn.Module):
    def __init__(self,
                 ntoken,
                 emb_size,
                 embs_dropout,
                 embs_fixed=False,
                 pretrain_embs=None):
        super(TERBase_, self).__init__()
        ## embedding ##
        pretrain_embs = torch.from_numpy(pretrain_embs).float() if pretrain_embs is not None else None
        self.embedding = nn.Embedding(ntoken, emb_size, padding_idx=ntoken-1, _weight=pretrain_embs)
        if type(pretrain_embs) == type(None):
            if embs_fixed:
                print("Warning: Fix randomly initialized embedding matrix. "
                      "Perhaps you should provide the argument pretrain_emb?")
            print ("Randomly initialize embedding matrix.")
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        else:
            print ("Embedding matrix is pretrained.")
        print("Embedding shape: %s" % str(self.embedding.weight.size()))
        if embs_fixed:
            self.embedding.weight.requires_grad = False
        print("Fixed embedding: %s" % str(not self.embedding.weight.requires_grad))
        # emb dropout
        self.emb_dropout = nn.Dropout(embs_dropout) if embs_dropout > 0 else None
        ## attributes ##
        self.ntoken = ntoken

    def forward(self, inputs, lengths):
        raise NotImplemented


class TER_Avg_Encoder(TERBase_):
    def __init__(self,
                 ntoken,
                 emb_size,
                 embs_dropout,
                 embs_fixed=False,
                 pretrain_embs=None):
        super(TER_Avg_Encoder, self).__init__(ntoken,
                                              emb_size,
                                              embs_dropout=embs_dropout,
                                              embs_fixed=embs_fixed,
                                              pretrain_embs=pretrain_embs)

    def forward(self, inputs, lengths):
        lengths = torch.tensor(lengths, dtype=torch.long, device=inputs.device)
        inputs_emb = self.embedding(inputs)
        if self.emb_dropout:
            inputs_emb = self.emb_dropout(inputs_emb)
        emb_avgs = torch.sum(inputs_emb, dim=1) / lengths[:, None].float()
        return emb_avgs


class TER_FFNN(ERBase_):
    def __init__(self,
                 ntoken,
                 emb_size,
                 nhid,
                 class_num,
                 nlayers,
                 dropout,
                 embs_dropout,
                 embs_fixed=False,
                 pretrain_embs=None):
        super(TER_FFNN, self).__init__(emb_size, nhid, class_num, fc_nlayers=nlayers, fc_dropout=dropout)
        self.ter_avg_encoder = TER_Avg_Encoder(ntoken, emb_size, embs_dropout, embs_fixed, pretrain_embs)

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        outs = self.ter_avg_encoder(token_ids, tok_lengths)
        outs = self.fcn(outs)
        return outs
        

class TER_RNN_Encoder(TERBase_):
    def __init__(self,
                 cell_type,
                 ntoken,
                 emb_size,
                 ninp,
                 nhid,
                 nlayers,
                 dropout,
                 bidirection,
                 embs_dropout,
                 embs_fixed=False,
                 pretrain_embs=None):
        if not emb_size == ninp:
            raise ValueError("In RNN, emb_size should be equal to ninp, but receive %d, %d" % (emb_size, ninp))
        # embeddings
        super(TER_RNN_Encoder, self).__init__(ntoken,
                                              emb_size,
                                              embs_dropout=embs_dropout,
                                              embs_fixed=embs_fixed,
                                              pretrain_embs=pretrain_embs)
        # rnn
        rnn_kwargs = {'input_size': ninp,
                      'hidden_size': nhid,
                      'num_layers': nlayers,
                      'batch_first': True,
                      'dropout': dropout,
                      'bidirectional': bidirection}
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(**rnn_kwargs)
        elif cell_type in ['RNN_RELU', 'RNN_TANH']:
            self.rnn = nn.RNN(**rnn_kwargs, nonlinearity=cell_type[4:].lower())
        else:
            raise ValueError("Unsupported rnn cell type: %s" % cell_type)
        self.enc_dropout = nn.Dropout(dropout) if dropout > 0 else None
        # attributes
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nhid * (bidirection + 1)

    def forward(self, inputs, lengths):
        inputs_emb = self.embedding(inputs)
        #inputs_emb = inputs_emb * self.position_encoding
        if self.emb_dropout:
            inputs_emb = self.emb_dropout(inputs_emb)
        # make packed padded sequences
        lengths = torch.tensor(lengths, dtype=torch.long, device=inputs.device)
        sorted_lengths, sorted_idxs = torch.sort(lengths, descending=True)
        _, unsorted_idxs = torch.sort(sorted_idxs)
        inputs_emb = inputs_emb[sorted_idxs, :, :]
        packed_inputs_emb = nn.utils.rnn.pack_padded_sequence(inputs_emb, sorted_lengths, batch_first=True)
        # rnn
        rnn_outs, hiddens = self.rnn(packed_inputs_emb)
        # unpacked
        #rnn_outs = nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)[0]
        #rnn_outs = rnn_outs[unsorted_idxs, :, :]
        #temp = rnn_outs
        # get final outputs
        #rnn_outs = rnn_outs[torch.arange(rnn_outs.size(0)), inputs_seq_len-1, :]
        if type(hiddens) == tuple:
            hiddens = hiddens[0]
        rnn_outs = hiddens.permute(1, 0, 2)[unsorted_idxs, :, :]
        rnn_outs = rnn_outs.view(-1, self.nlayers, self.nout)[:, -1, :]
        if self.enc_dropout:
            rnn_outs = self.enc_dropout(rnn_outs)
        return rnn_outs


class TER_RNN(ERBase_):
    def __init__(self,
                 cell_type,
                 ntoken,
                 emb_size,
                 ninp,
                 nhid,
                 nlayers,
                 dropout,
                 bidirection,
                 class_num,
                 fc_nhid,
                 fc_nlayers,
                 fc_dropout,
                 embs_dropout,
                 embs_fixed=False,
                 pretrain_embs=None):
        if not emb_size == ninp:
            raise ValueError("In RNN, emb_size should be equal to ninp, but receive %d, %d" % (emb_size, ninp))
        ter_rnn_encoder = TER_RNN_Encoder(cell_type,
                                          ntoken,
                                          emb_size,
                                          ninp,
                                          nhid,
                                          nlayers,
                                          dropout,
                                          bidirection,
                                          embs_dropout,
                                          embs_fixed,
                                          pretrain_embs)
        super(TER_RNN, self).__init__(ter_rnn_encoder.nout,
                                      fc_nhid,
                                      class_num,
                                      fc_nlayers,
                                      fc_dropout)
        self.ter_rnn_encoder = ter_rnn_encoder

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        outs = self.ter_rnn_encoder(token_ids, tok_lengths)
        outs = self.fcn(outs)
        return outs

