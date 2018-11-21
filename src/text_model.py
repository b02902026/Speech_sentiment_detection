import numpy as np
import torch
from torch import nn


class FullyConnectedNN(nn.Module):
    nolinearity = {'relu': nn.ReLU,
                   'elu': nn.ELU,
                   'leaky_relu': nn.LeakyReLU,
                   'prelu': nn.PReLU,
                   'selu': nn.SELU,
                   'sigmoid': nn.Sigmoid,
                   'tanh': nn.Tanh}
    def __init__(self, ninp, nhid, nout, nlayers, nonlinear, dropout):
        super(FullyConnectedNN, self).__init__()
        self.layers = [nn.Linear(ninp, nhid)] + \
                      [nn.Linear(nhid, nhid) for i in range(nlayers-2)] + \
                      [nn.Linear(nhid, nout)] if nlayers > 1 else [nn.Linear(ninp, nout)]
        for i in range(len(self.layers)):
            self.add_module("layer_%d" % i, self.layers[i])
        self.nonlinear = self.nolinearity[nonlinear]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        ## init ##
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, inputs):
        outs = inputs
        for i in range(len(self.layers)-1):
            outs = self.nonlinear(self.layers[i](outs))
            if self.dropout:
                outs = self.dropout(outs)
        outs = self.layers[-1](outs)
        return outs


class TERBase_(nn.Module):
    def __init__(self,
                 ntoken,
                 emb_size,
                 fc_ninp,
                 fc_nhid,
                 nout,
                 fc_nlayers,
                 fc_nonlinear,
                 fc_dropout,
                 embs_dropout,
                 embs_fixed=False,
                 sparse=False,
                 pretrain_embs=None):
        super(TERBase_, self).__init__()
        ## feed forward net ##
        self.fcn = FullyConnectedNN(fc_ninp, fc_nhid, nout, fc_nlayers, fc_nonlinear, fc_dropout)
        ## embedding ##
        pretrain_embs = torch.from_numpy(pretrain_embs).float() if pretrain_embs is not None else None
        self.embedding = nn.Embedding(ntoken, emb_size, padding_idx=ntoken-1, sparse=sparse, _weight=pretrain_embs)
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
        self.enc_dropout = nn.Dropout(fc_dropout) if fc_dropout > 0 else None
        self.emb_dropout = nn.Dropout(embs_dropout) if embs_dropout > 0 else None
        ## attributes ##
        self.ntoken = ntoken

    def forward(self, inputs, lengths):
        raise NotImplemented


class TER_FFNN(TERBase_):
    def __init__(self,
                 ntoken,
                 emb_size,
                 ninp,
                 nhid,
                 nout,
                 nlayers,
                 nonlinear,
                 dropout,
                 embs_dropout,
                 embs_fixed=False,
                 sparse=False,
                 pretrain_embs=None):
        if not emb_size == ninp:
            raise ValueError("In FFNN, emb_size should be equal to ninp, but receive %d, %d" % (emb_size, ninp))
        super(TER_FFNN, self).__init__(ntoken,
                                            emb_size,
                                            ninp,
                                            nhid,
                                            nout,
                                            nlayers,
                                            nonlinear,
                                            dropout,
                                            embs_dropout=embs_dropout,
                                            embs_fixed=embs_fixed,
                                            sparse=sparse,
                                            pretrain_embs=pretrain_embs)

    def forward(self, inputs, lengths):
        lengths = torch.tensor(lengths, dtype=torch.long, device=inputs.device)
        inputs_emb = self.embedding(inputs)
        if self.emb_dropout:
            inputs_emb = self.emb_dropout(inputs_emb)
        emb_avgs = torch.sum(inputs_emb, dim=1) / lengths[:, None].float()
        outs = self.fcn(emb_avgs)
        return outs
        

class TER_RNN(TERBase_):
    def __init__(self,
                 cell_type,
                 ntoken,
                 emb_size,
                 ninp,
                 nhid,
                 nlayers,
                 dropout,
                 bidirection,
                 nout,
                 fc_nhid,
                 fc_nlayers,
                 fc_nonlinear,
                 fc_dropout,
                 embs_dropout,
                 embs_fixed=False,
                 sparse=False,
                 pretrain_embs=None):
        if not emb_size == ninp:
            raise ValueError("In RNN, emb_size should be equal to ninp, but receive %d, %d" % (emb_size, ninp))
        # fcn and embeddings
        super(TER_RNN, self).__init__(ntoken,
                                           emb_size,
                                           nhid * (bidirection + 1),
                                           fc_nhid,
                                           nout,
                                           fc_nlayers,
                                           fc_nonlinear,
                                           fc_dropout,
                                           embs_dropout=embs_dropout,
                                           embs_fixed=embs_fixed,
                                           sparse=sparse,
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
        # attributes
        self.nhid = nhid
        self.nlayers = nlayers
        self.bidirection = bidirection

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
        rnn_outs = rnn_outs.view(-1, self.nlayers, (self.bidirection+1)*self.nhid)[:, -1, :]
        if self.enc_dropout:
            rnn_outs = self.enc_dropout(rnn_outs)
        # fcn
        outs = self.fcn(rnn_outs)
        #return outs, temp
        return outs

