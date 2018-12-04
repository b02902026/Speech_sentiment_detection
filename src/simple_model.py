import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import ERBase_

### SER RNN Model ###
class SER_RNN_Encoder(nn.Module):
    def __init__(self, h_size, feat_size, dropout=0):
        super(SER_RNN_Encoder, self).__init__()
        self.hidden_size = h_size
        self.input_size = feat_size
        self.transform = nn.Linear(self.input_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        ## attributes
        self.nout = 2 * h_size

    def forward(self, x, lengths):
        B = x.size(0)
        #x = self.transform(x) # (B, S, H)
        x = self.tanh(x)
        pack = pack_padded_sequence(x, lengths, batch_first=True)
        x, (h, c) = self.rnn(pack)
        if self.dropout:
            h = self.dropout(h)
        out = h.permute(1,0,2).contiguous().view(B,-1)
        
        return out


class SER(ERBase_):
    def __init__(self, h_size, feat_size, class_num, dropout=0):
        super(SER, self).__init__(h_size*2, h_size, class_num, fc_nlayers=1, fc_dropout=dropout)
        self.ser_rnn_encoder = SER_RNN_Encoder(h_size, feat_size, dropout)

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        outs = self.ser_rnn_encoder(speech_inputs, speech_lengths)
        outs = self.fcn(outs)
        return outs




### SER CNN Model ###
class ExtendConv(nn.Module):
    """ Special Conv that mix multiple filter with different kernel(strides...) sizes
        len(kernels) should be equal to len(strides), len(dilations), len(groups)
    """
    def __init__(self, conv_type, ninp, nfilter, kernels, strides, dilations, groups):
        super(ExtendConv, self).__init__()
        # check args
        self.check_args(kernels, strides, dilations, groups)
        # conv modules
        if conv_type == '1d':
            block = nn.Conv1d
        elif conv_type == '2d':
            block = nn.Conv2d
        self.convs = []
        for i in range(len(kernels)):
            m = block(ninp, nfilter, kernels[i], strides[i], (kernels[i]-1)//2, dilations[i], groups[i])
            self.convs.append(m)
        for i in range(len(kernels)):
            self.add_module("conv_%d" % i, self.convs[i])
        self.nout = len(kernels)*nfilter

    def check_args(self, *args):
        m_num = len(args[0])
        for arg in args[1:]:
            if not len(arg) == m_num:
                raise ValueError("length of each arg(kernels, strides, ...) should be equal")

    def forward(self, inputs):
        """
        inputs
          shape: [B, C, T, ninp] or [B, ninp, T]

        outs:
          shape: [B, nout, T, ninp] or [B, nout, T]
        """
        outs = [conv(inputs) for conv in self.convs]
        if len(self.convs) > 1:
            outs = torch.cat(outs, dim=1)
        else:
            outs = outs[0]
        return outs


class SER_CNN_Encoder(nn.Module):
    def __init__(self,
                 conv_type,
                 nfilter,
                 feat_size,
                 max_time_step,
                 nlayers,
                 kernel,
                 stride=1,
                 dilation=1,
                 group=1,
                 dropout=0.):
        super(SER_CNN_Encoder, self).__init__()
        # modify argumants
        stride = self.get_layer_arg(kernel, stride)
        dilation = self.get_layer_arg(kernel, dilation)
        group = self.get_layer_arg(kernel, group)
        # cnn layers
        self.cnn_layers = []
        in_out = [1, nfilter]
        for i in range(nlayers):
            conv = ExtendConv(conv_type, *in_out, kernel, stride, dilation, group)
            self.cnn_layers.append(conv)
            in_out = [conv.nout, nfilter]
        for i in range(nlayers):
            self.add_module("cnn_layer_%d" % i, self.cnn_layers[i])
        # other modules
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        ## attributs ##
        if conv_type == '2d':
            nout = feat_size // (2**nlayers) * max_time_step // (2**nlayers) * nfilter
        elif conv_type == '1d':
            nout = max_time_step // (2**nlayers) * nfilter
        self.nout = nout
        self.conv_type = conv_type
        ## init ##
        self.reset_parameters()

    def get_layer_arg(self, kernel, arg):
        if type(kernel) == int:
            if not type(arg) == int:
                raise ValueError("Arg should be type int when kernel is type int!")
        elif type(kernel) == list:
            if type(arg) == list:
                if not len(arg) == len(kernel):
                    raise ValueError("Arg should have the same length with kernel when they are both type list!")
            elif type(arg) == int:
                arg = [arg]*len(kernel)
            else:
                raise TypeError("arg should be int or list!")
        else:
            raise TypeError("kernel should be int or list!")
        return arg

    def reset_parameters(self):
        def init_weight(m):
            if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
                nn.init.xavier_normal_(m.weight)
        self.apply(init_weight)

    def forward(self, inputs, lengths):
        """ The dropout is not applied to the last layer
        inputs:
          shape: [B, T, ninp]

        outs:
          shape: [B, class_num]
        """
        if self.conv_type == '2d':
            outs = inputs.unsqueeze(1) # shape: [B, 1, T, ninp]
            pool_func = F.max_pool2d
        elif self.conv_type == '1d':
            outs = inputs.permute(0, 2, 1)
            pool_func = F.max_pool1d
        for i in range(len(self.cnn_layers)):
            outs = self.nonlinear(self.cnn_layers[i](outs))
            outs = pool_func(outs, kernel_size=2)
            if self.dropout:
                outs = self.dropout(outs)
        outs = outs.view(inputs.size(0), self.nout)
        return outs


class SER_CNN(ERBase_):
    def __init__(self,
                 conv_type,
                 h_size,
                 feat_size,
                 class_num,
                 max_time_step,
                 nlayers,
                 kernel,
                 stride=1,
                 dilation=1,
                 group=1,
                 dropout=0.):
        ser_cnn_encoder = SER_CNN_Encoder(conv_type,
                                          h_size,
                                          feat_size,
                                          max_time_step,
                                          nlayers,
                                          kernel,
                                          stride,
                                          dilation,
                                          group,
                                          dropout)
        super(SER, self).__init__(ser_cnn_encoder.nout, h_size, class_num, fc_nlayers=1, fc_dropout=dropout)
        self.ser_cnn_encoder = ser_cnn_encoder

    def forward(self, speech_inputs, speech_lengths, token_ids, tok_lengths):
        outs = self.ser_cnn_encoder(speech_inputs, speech_lengths)
        outs = self.fcn(outs)
        return outs
