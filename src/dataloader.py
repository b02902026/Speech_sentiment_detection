import torch as th
from torch.utils.data import Dataset, DataLoader
import json, pickle
import librosa
import numpy as np


def normalize(data, axis):
    return (data - np.mean(data, axis=axis, keepdims=True)) / np.std(data, axis=axis, keepdims=True)


def get_mean_var(all_data):
    '''
    all_data: List[exp, ...]
      exp: shape : [T, feat_size]
    '''
    all_data = np.concatenate(all_data, axis=0)
    mean_data = np.mean(all_data, axis=0, keepdims=True)
    std_data = np.std(all_data, axis=0, keepdims=True)
    return mean_data, std_data


class IEMOCAP(Dataset):
    def __init__(self, data_path, feat_size, pad_type):
        #with open(data_path, 'r') as f:
        #    self.data = json.load(f)
        self.feat_size = feat_size
        self.pad_type = pad_type
        assert self.pad_type in ['batch', 'global']
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        all_mfcc = []
        all_chroma = []
        all_rmse = []
        all_zero_crossing = []
        all_audio = []
        all_raw = []

        max_s = 0
        # process raw wav
        if 'raw' in self.data[0]:
            for d in self.data:
                # pad the las one
                #print(d['raw'].shape, d['raw'][0].shape,d['raw'][1].shape, d['raw'][-1].shape)
                d['raw'][-1]  = np.concatenate((d['raw'][-1], np.zeros((200-d['raw'][-1].shape[0]))), axis=None)
                d['raw'] = np.stack(d['raw']).reshape(-1, 200)
                all_raw.append(d['raw'])
            mean_raw, std_raw = get_mean_var(all_raw)
            for d in self.data:
                d['raw'] = (d['raw'] - mean_raw) / std_raw
                max_s = max(max_s, d['raw'].shape[0])
        
        # process others
        if 'mfcc' in self.data[0]:
            for d in self.data:
                all_mfcc.append(d['mfcc'])
                all_chroma.append(d['chroma'])
                all_rmse.append(d['rmse'])
                all_zero_crossing.append(d['zero_crossing'])
                #all_audio.append(d['audio'])
                all_audio.append(np.concatenate((d['audio'][:, :8],d['audio'][:, 21:]), axis=1))


            mean_mfcc, std_mfcc = get_mean_var(all_mfcc)
            mean_chroma, std_chroma = get_mean_var(all_chroma)
            mean_rmse, std_rmse = get_mean_var(all_rmse)
            mean_zero_crossing, std_zero_crossing = get_mean_var(all_zero_crossing)
            mean_audio, std_audio = get_mean_var(all_audio)
            
            for d in self.data:
                d['mfcc'] = (d['mfcc'] - mean_mfcc) / std_mfcc
                d['chroma'] = (d['chroma'] - mean_chroma) / std_chroma
                d['rmse'] = (d['rmse'] - mean_rmse) / std_rmse
                d['zero_crossing'] = (d['zero_crossing'] - mean_zero_crossing) / std_zero_crossing
                #d['audio'] = (d['audio'] - mean_audio) / std_audio
                d['audio'] = (np.concatenate((d['audio'][:, :8],d['audio'][:, 21:]), axis=1) - mean_audio) / std_audio
                
                max_s = max(max_s, d['mfcc'].shape[0])
        

        # TODO: may need to modify
        #self.max_time_step = 736
        #self.max_au_time_step = 1063
        #self.max_time_step = max([x.shape[0] for x in all_mfcc])
        #self.max_time_step = max([x.shape[0] for x in all_audio])
        '''
        if self.feat_size in [54, 40]:
            self.max_time_step = 736
        elif self.feat_size == 34:
            self.max_time_step = 1063
        '''
        #self.max_time_step = 1063
        self.max_time_step = 1882
        #self.max_time_step = 1471

    
    def __getitem__(self, index):
        '''
        y, sr = librosa.load(self.data[index]['wav_path'], sr=11025)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12).T
        rmse = librosa.feature.rmse(y).T
        zero_crossing = librosa.feature.zero_crossing_rate(y).T
        speech_feats = np.concatenate([mfcc, chroma, rmse, zero_crossing], axis=-1)
        '''
        if self.feat_size == 54:
            speech_feats = np.concatenate([self.data[index]['mfcc'],
                                           self.data[index]['chroma'],
                                           self.data[index]['rmse'],
                                           self.data[index]['zero_crossing']
                                           ], axis=-1)

        elif self.feat_size == 40:
            speech_feats = self.data[index]['mfcc']
        elif self.feat_size == 34:
            speech_feats = self.data[index]['audio']
        elif self.feat_size == -1:   # use raw signal input
            speech_feats = self.data[index]['raw']
                       
        label = self.data[index]['category_index']
        #if "text" in self.data[index]:
        #    text = self.data[index]['text']
        #else:
        #    text = ""
        tokens_id = self.data[index]['tokens_id']
        #return mfcc, label, tokens_id
        return speech_feats, label, tokens_id
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, b):
        b = sorted(b, key=lambda x:x[0].shape[0], reverse=True)
        speech_feats, label, tokens_id = map(list,zip(*b))
        # speech feats padding
        speech_length = [x.shape[0] for x in speech_feats]
        speech_max_len = speech_length[0] if self.pad_type == 'batch' else self.max_time_step
        for i, feat in enumerate(speech_feats):
            speech_feats[i] = np.concatenate((feat, np.zeros((speech_max_len-feat.shape[0], feat.shape[1]))), axis=0)
        # tokens_id padding
        tok_length = [len(x) for x in tokens_id]
        tok_max_len = max(tok_length)
        for i, token_id in enumerate(tokens_id):
            tokens_id[i] += [0]*(tok_max_len - len(token_id))
        
        return th.FloatTensor(speech_feats), speech_length, th.LongTensor(label), th.LongTensor(tokens_id), tok_length

def get_dataloader(path, batch_size, shuffle, feat_size, pad_type):
    IEMO = IEMOCAP(path, feat_size, pad_type)
    return DataLoader(IEMO, batch_size=batch_size, shuffle=shuffle, collate_fn=IEMO.collate_fn)
    


