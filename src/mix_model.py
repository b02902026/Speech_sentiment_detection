import numpy as np
import torch
from torch import nn
from model import ERBase_
from text_model import TER_RNN_Encoder, TER_Avg_Encoder
from simple_model import SER_RNN_Encoder, SER_CNN_Encoder


class MixER(ERBase_):
    def __init__(self,
                 speech_feat_size,
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
            raise ValueError("In Text RNN, emb_size should be equal to ninp, but receive %d, %d" % (emb_size, ninp))
        ser_rnn_encoder = SER_RNN_Encoder(nhid, speech_feat_size, dropout)
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
        fc_ninp = ser_rnn_encoder.nout + ter_rnn_encoder.nout
        super(MixER, self).__init__(fc_ninp,
                                    fc_nhid,
                                    class_num,
                                    fc_nlayers,
                                    fc_dropout)
        self.ser_rnn_encoder = ser_rnn_encoder
        self.ter_rnn_encoder = ter_rnn_encoder

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        ser_outs = self.ser_rnn_encoder(speech_inputs, speech_lengths)
        ter_outs = self.ter_rnn_encoder(token_ids, tok_lengths)
        outs = torch.cat([ser_outs, ter_outs], dim=-1)
        outs = self.fcn(outs)
        return outs

