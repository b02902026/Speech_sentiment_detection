import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from simple_model import SER_RNN_Encoder

class BreathClassifier(nn.Module):
    def __init__(self, h_size, kernel_size, class_num, mix = False, ser_hidden = 200, ser_feat=40):
        super(BreathClassifier, self).__init__()
        self.hidden_size = h_size
        self.kernel_size = kernel_size
        self.mix = mix
        self.ser_encoder = SER_RNN_Encoder(h_size=ser_hidden, feat_size=ser_feat, dropout=0.2)
        convs = []
        for i, k in enumerate(kernel_size):
            cnn = nn.Conv1d(1, self.hidden_size, k)
            convs.append(cnn)
        
        self.conv_block = nn.ModuleList(convs)
        self.fc = nn.Linear(self.hidden_size * len(self.kernel_size), class_num)
        self.merger_fc = nn.Linear(ser_hidden * 2 + self.hidden_size * len(self.kernel_size), class_num)
    
    def forward(self, breath, wav_feat, wav_feat_length):
        
        x = breath.unsqueeze(1) # (B, 1, S)
        seq_len = x.size(-1)
        outs = []
        for i in range(len(self.conv_block)):
            out = self.conv_block[i](x) # (B, H, S)
            out = F.max_pool1d(out, kernel_size=seq_len-self.kernel_size[i]+1) # (B, H, 1)
            outs.append(out)
        
        outs = th.cat(outs, dim=2)  # (B, H, # of kernels)
        outs = outs.view(-1, self.hidden_size * len(self.kernel_size)) # (B, H * # of kernel)
        # combine wav feature
        if self.mix:
            encoder_out = self.ser_encoder(wav_feat, wav_feat_length) 
            pred = self.merger_fc(th.cat([encoder_out, outs], dim=-1)) 
        else:
            pred = self.fc(outs)
        return pred
    



