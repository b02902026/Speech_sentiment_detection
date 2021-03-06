import argparse
import os
import json
import pickle
import librosa
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
import collections
import spacy
import numpy as np
from pydub import AudioSegment
import random

PREFIX = '../data'
nlp = spacy.blank('en')

def load_embedding(vocab, path, dimension=300):
    print('loading glove {} d'.format(dimension))
    embed = np.random.normal(0, 1, size=(len(vocab), dimension))
    embed[vocab('<pad>')] = np.zeros((dimension))
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            tmp = line.strip().split()
            word = tmp[0]
            vec = list(map(float, tmp[1:]))
            if word in vocab.word2idx:
                embed[vocab(word)] = vec
    
    with open(os.path.join(PREFIX, "glove.6B.%dd-relativized.pkl" % dimension), 'wb') as f:
        pickle.dump(embed, f)

def tokenize(s):
    s = s.lower()
    doc = nlp(s)
    tokens = [token.text for token in doc]
    return tokens

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']

    def add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def build(self, counter, threshold=1):
        for w, count in counter.items():
            if count >= threshold:
                self.add(w)
    
    def __len__(self):
        return len(self.word2idx)

def get_labels(file_dir, label_dict):
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        if os.path.isdir(file_path):
            continue
        with open(file_path, 'r') as f:
            for line in f:
                if line[0] != '[':
                    continue
                prefix = file_name.split('.')[0]
                start = line.find(prefix)
                if start == -1:
                    raise Exception("sth goes weong...")
                line = line[start:]
                name = line.split()[0]
                category_label = line.split()[1]
                #start = line.find('[')
                #line = line[start+1:-1]
                #dimensional_label = line.split()
                label_dict[name] = {"category":category_label}

def get_text(file_dir, d, w_counter):
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                name = line.split('[')[0].strip()
                text = line.split(':')[1].strip()
                try:
                    tokens = tokenize(text)
                    d[name]['text'] = text
                    d[name]['text_tokens'] = tokens
                    w_counter.update(tokens)
                except KeyError:
                    print("Can't parse {}".format(name))

def get_wav_feature(file_dir, wav_dict):
    for dir_name in os.listdir(file_dir):
        sec_dir = os.path.join(file_dir, dir_name)
        for file_name in os.listdir(sec_dir):
            file_path = os.path.join(sec_dir, file_name)
            if file_path.endswith('.wav'):
                #y, sr = librosa.load(file_path)
                #mfcc = librosa.feature.mfcc(y=y, sr=sr)
                name = file_name.split('.')[0]
                #wav_dict[name]["feat"] = mfcc
                wav_dict[name]["wav_path"] = file_path
            else:
                print("{} fail".format(file_path))

def get_gathered_breath(file_dir, data_dict):
    # get breath 
    for filename in os.listdir(file_dir):
        with open(os.path.join(file_dir,filename)) as f:
            lines = f.readlines()[7:-5]
        
        name = filename.split('.')[0]
        # can change the granularity of time
        breath_wave = []
        interval_size = 1
        for i, l in enumerate(lines):
            time, value = l.strip().split(',')
            if i % interval_size == 0:
                breath_wave.append(int(value))

        data_dict[name]['breath'] = breath_wave

def get_gathered_wav(file_dir, data_dict):
    # get self-recoreded wav file name
    for filename in os.listdir(file_dir):
        name = filename.split('.')[0]
        mysound = AudioSegment.from_wav(os.path.join(file_dir, filename))
        mysound = mysound.set_channels(1)
        mysound.export(os.path.join(file_dir, name+ '_mono.wav'), format="wav")
        data_dict[name]['wav_path'] = os.path.join(file_dir, name+ '_mono.wav')


def recategorize_and_split(json_path):
    #class_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'fru': 4, 'exc': 1, 'fea': 4, 'sur': 4, 'dis': 4, 'oth': 4, 'xxx': 4}
    class_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 1}
    with open(json_path, 'rb') as f:
        data = pickle.load(f)

    counter = {}
    #class_map = {'ang':0, 'exc':1, 'sur':1, 'fru':2, 'hap':3, 'sad':2}
    class_num = max(class_map.values()) + 1
    class_count = [0 for _ in range(class_num)]
    small_data = []
    train_data = []
    val_data = []
    for instance in data:
        session_id = int(instance['wav_path'].split('/')[-1].split('_')[0][4])
        cat = instance['category']
        if cat not in counter and cat in class_map:
            counter[cat] = 1
        elif cat in class_map:
            counter[cat] += 1

        if cat in class_map:
            instance['category_index'] = class_map[cat]
            '''
            if session_id == 5:
                val_data.append(instance)
            else:
                train_data.append(instance)
            '''
            small_data.append(instance)
            class_count[class_map[cat]] += 1

    train_size = int(len(small_data) * 0.8)
    #random.shuffle(small_data)
    print("statistics:")
    print(counter)
    print("original size: ", len(data))
    print("train size: ", len(train_data))
    print("val size: ", len(val_data))
    weight = 1 / np.asarray(class_count)
    print(weight / np.sum(weight))
    '''
    with open('../data/train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('../data/val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    '''
    with open('../data/train.pkl', 'wb') as f:
        pickle.dump(small_data[:train_size], f)
    with open('../data/val.pkl', 'wb') as f:
        pickle.dump(small_data[train_size:], f)
    return class_num

def extract_from_wav(wav_file, rate=11025):
    y, sr = librosa.load(wav_file, sr=rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--breath', action='store_true', default=False)
    parser.add_argument('--gather_path', type=str, default="../data/IEMOCAP_gather")
    parser.add_argument('-output', type=str, default="../data/data.pkl")
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('-sr', type=int, default=11025)
    args = parser.parse_args()
    # Do the breath experiment
    w_counter = collections.Counter()
    vocab = Vocabulary()
    exp_dict = {}
    SESS_N = 5
    for i in range(1, SESS_N+1):
        get_labels('../data/IEMOCAP_full_release_light/Session{}/dialog/EmoEvaluation'.format(i), exp_dict)
        get_wav_feature('../data/IEMOCAP_full_release_light/Session{}/sentences/wav'.format(i), exp_dict)
        get_text('../data/IEMOCAP_full_release_light/Session{}/dialog/transcriptions'.format(i), exp_dict, w_counter)
    
    # filter
    if args.breath:
        get_gathered_breath(os.path.join(args.gather_path, 'breath'), exp_dict)
        get_gathered_wav(os.path.join(args.gather_path, 'wav'), exp_dict)
        subset_dict = {}
        for k, v in exp_dict.items():
            if 'breath' in v:
                subset_dict[k] = v

        exp_dict = subset_dict
        print("breath training size is ", len(exp_dict))
    

    print("Build Vocabulary...")
    vocab.build(w_counter)
    print("%d words in total" % len(vocab))

    #with open('examples.pkl', 'wb') as f:
    #    pickle.dump(exp_dict, f)
    exps = []
    for k, v in exp_dict.items():
        exp_dict[k]['id'] = k
        exp_dict[k]['tokens_id'] = list(map(vocab, exp_dict[k]['text_tokens']))
        # librosa
        y, sr = librosa.load(exp_dict[k]['wav_path'], sr=args.sr)
        # use raw
        if args.raw:
            frames = []
            window_size = 100
            hop_size = 0
            for t in range(0, len(y), window_size):
                frames.append(y[t:t+wondow_size])

            exp_dict[k]['raw'] = np.asarray(frames)
            exps.append(exp_dict[k])
            continue

        # features from librosa
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12).T
        rmse = librosa.feature.rmse(y).T
        zero_crossing = librosa.feature.zero_crossing_rate(y).T
        # pyAudioAnalysis
        [Fs, y] = audioBasicIO.readAudioFile(exp_dict[k]['wav_path'])
        #all_feats, f_names = audioFeatureExtraction.stFeatureExtraction(y, Fs, int(0.025*Fs), int(0.010*Fs))
        all_feats, f_names = audioFeatureExtraction.stFeatureExtraction(y, Fs, 2048, 512)
        exp_dict[k]['mfcc'] = mfcc
        exp_dict[k]['chroma'] = chroma
        exp_dict[k]['rmse'] = rmse
        exp_dict[k]['zero_crossing'] = zero_crossing
        exp_dict[k]['audio'] = all_feats.T
        exps.append(exp_dict[k])

    #with open('../data/data.json', 'w') as f:
    #    json.dump(exps, f, indent=4)
    with open(args.output, 'wb') as f:
        pickle.dump(exps, f)

    #recategorize_and_split('../data/data.json')
    with open('../data/vocab.pkl','wb') as f:
        pickle.dump(vocab, f)
    '''
    load_embedding(vocab, '../data/glove.6B.50d.txt', 50)
    load_embedding(vocab, '../data/glove.6B.100d.txt', 100)
    load_embedding(vocab, '../data/glove.6B.200d.txt', 200)
    load_embedding(vocab, '../data/glove.6B.300d.txt', 300)
    '''
