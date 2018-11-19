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
        y, sr = librosa.load(self.data[index]['wav_path'])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        label = self.data[index]['category']
        if "text" in self.data[index]:
            text = self.data[index]['text']
        else:
            text = ""
        return mfcc, label, text
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, b):
        b = sorted(b, key=lambda x:x[0].shape[1], reverse=True)
        mfcc, label, text = map(list,zip(*b))
        length = [x.shape[1] for x in mfcc]
        max_len = length[0]
        for i, feat in enumerate(mfcc):
            mfcc[i] = np.concatenate((feat, np.zeros((40, max_len-feat.shape[1]))), axis=1)
        
        return th.FloatTensor(mfcc).transpose(1,2), length, th.LongTensor(label), text

def get_dataloader(path, batch_size, shuffle):
    IEMO = IEMOCAP(path)
    return DataLoader(IEMO, batch_size=batch_size, shuffle=shuffle, collate_fn=IEMO.collate_fn)
    


