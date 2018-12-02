import argparse
import os
import json
import pickle
import librosa
import collections
import spacy
import numpy as np

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


def recategorize_and_split(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    class_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'fru': 4, 'exc': 1, 'fea': 4, 'sur': 4, 'dis': 4, 'oth': 4, 'xxx': 4}
    class_count = [0,0,0,0,0]
    counter = {}
    small_data = []
    for instance in data:
        cat = instance['category']
        if cat not in counter:
            counter[cat] = 1
        else:
            counter[cat] += 1

        if cat in class_map:
            instance['category_index'] = class_map[cat]
            small_data.append(instance)
            class_count[class_map[cat]] += 1

    train_size = int(len(small_data) * 0.9)
    #train_size = 20
    print("statistics:")
    print(counter)
    print("new train size: ", train_size)
    weight = 1 / np.asarray(class_count)
    print(weight / np.sum(weight))

    with open('../data/val.json', 'w+') as f:
        json.dump(small_data[train_size:], f, indent=4)
    
    return 5


if __name__ == "__main__":
    w_counter = collections.Counter()
    vocab = Vocabulary()
    exp_dict = {}
    SESS_N = 5
    for i in range(1, SESS_N+1):
        get_labels('../data/IEMOCAP_full_release_light/Session{}/dialog/EmoEvaluation'.format(i), exp_dict)
        get_wav_feature('../data/IEMOCAP_full_release_light/Session{}/sentences/wav'.format(i), exp_dict)
        get_text('../data/IEMOCAP_full_release_light/Session{}/dialog/transcriptions'.format(i), exp_dict, w_counter)
    print("Build Vocabulary...")
    vocab.build(w_counter)
    print("%d words in total" % len(vocab))

    #with open('examples.pkl', 'wb') as f:
    #    pickle.dump(exp_dict, f)
    exps = []
    for k, v in exp_dict.items():
        exp_dict[k]['id'] = k
        exp_dict[k]['tokens_id'] = list(map(vocab, exp_dict[k]['text_tokens']))
        exps.append(exp_dict[k])

    with open('../data/data.json', 'w') as f:
        json.dump(exps, f, indent=4)

    recategorize_and_split('../data/data.json')
    '''
    with open('../data/vocab.pkl','wb') as f:
        pickle.dump(vocab, f)
    load_embedding(vocab, '../data/glove.6B.50d.txt', 50)
    load_embedding(vocab, '../data/glove.6B.100d.txt', 100)
    load_embedding(vocab, '../data/glove.6B.200d.txt', 200)
    load_embedding(vocab, '../data/glove.6B.300d.txt', 300)
    '''
