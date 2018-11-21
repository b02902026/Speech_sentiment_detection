import torch as th
from torch.utils.data import Dataset, DataLoader
import json
import librosa
import numpy as np

class IEMOCAP(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __getitem__(self, index):
        y, sr = librosa.load(self.data[index]['wav_path'], sr=11025)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        label = self.data[index]['category_index']
        #if "text" in self.data[index]:
        #    text = self.data[index]['text']
        #else:
        #    text = ""
        tokens_id = self.data[index]['tokens_id']
        return mfcc, label, tokens_id
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, b):
        b = sorted(b, key=lambda x:x[0].shape[0], reverse=True)
        #mfcc, label, text = map(list,zip(*b))
        mfcc, label, tokens_id = map(list,zip(*b))
        # mfcc padding
        mfcc_length = [x.shape[0] for x in mfcc]
        mfcc_max_len = mfcc_length[0]
        for i, feat in enumerate(mfcc):
            mfcc[i] = np.concatenate((feat, np.zeros((mfcc_max_len-feat.shape[0], 40))), axis=0)
        # tokens_id padding
        tok_length = [len(x) for x in tokens_id]
        tok_max_len = max(tok_length)
        for i, token_id in enumerate(tokens_id):
            tokens_id[i] += [0]*(tok_max_len - len(token_id))
        
        return th.FloatTensor(mfcc), mfcc_length, th.LongTensor(label), th.LongTensor(tokens_id), tok_length

def get_dataloader(path, batch_size, shuffle):
    IEMO = IEMOCAP(path)
    return DataLoader(IEMO, batch_size=batch_size, shuffle=shuffle, collate_fn=IEMO.collate_fn)
    


