import numpy as np
import torch
from torch import nn


class FullyConnectedNN(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout):
        super(FullyConnectedNN, self).__init__()
        self.layers = [nn.Linear(ninp, nhid)] + \
                      [nn.Linear(nhid, nhid) for i in range(nlayers-2)] + \
                      [nn.Linear(nhid, nout)] if nlayers > 1 else [nn.Linear(ninp, nout)]
        for i in range(len(self.layers)):
            self.add_module("layer_%d" % i, self.layers[i])
        self.nonlinear = nn.ReLU()
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


class ERBase_(nn.Module):
    def __init__(self,
                 fc_ninp,
                 fc_nhid,
                 nout,
                 fc_nlayers,
                 fc_dropout):
        super(ERBase_, self).__init__()
        ## feed forward net ##
        self.fcn = FullyConnectedNN(fc_ninp, fc_nhid, nout, fc_nlayers, fc_dropout)

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        raise NotImplemented
