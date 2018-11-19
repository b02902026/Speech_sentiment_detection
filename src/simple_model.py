import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SER(nn.Module):
    def __init__(self, h_size, feat_size, class_num):
        super(SER, self).__init__()
        self.hidden_size = h_size
        self.input_size = feat_size
        self.transform = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.output_fc = nn.Linear(self.hidden_size * 2, class_num)
        self.attention_parameter = nn.Parameter(th.zeros(self.hidden_size*2, self.hidden_size*2))
        th.nn.init.xavier_uniform_(self.attention_parameter)

    def forward(self, x, lengths):
        x = self.transform(x) # (B, S, H)
        pack = pack_padded_sequence(x, lengths, batch_first=True)
        x, (h, c) = self.rnn(pack)
        out = self.output_fc(h.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2))
        
        return out
    


