import argparse
import os
import json
import librosa
import pickle

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
                line = line[start:]
                name = line.split()[0]
                category_label = line.split()[1]
                start = line.find('[')
                line = line[start+1:-1]
                dimensional_label = line.split()
                label_dict[name] = {"category":category_label, "dimensional":dimensional_label}

def get_text(file_dir, d):
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                name = line.split('[')[0].strip()
                text = line.split(':')[1].strip()
                try:
                    d[name]['text'] = text
                except KeyError:
                    print("Can't parse {}".format(name))

def get_wav_feature(file_dir, wav_dict):
    for dir_name in os.listdir(file_dir):
        sec_dir = os.path.join(file_dir, dir_name)
        for file_name in os.listdir(sec_dir):
            file_path = os.path.join(sec_dir, file_name)
            if not file_path.endswith('.pk'):
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

    class_map = {'ang':0, 'exc':1, 'fru':2, 'hap':3, 'neu':4, 'sad':5}
    counter = {}
    small_data = []
    for instance in data:
        cat = instance['category']
        if cat not in counter:
            counter[cat] = 1
        else:
            counter[cat] += 1

        if cat in class_map:
            instance['category'] = class_map[cat]
            small_data.append(instance)

    print(counter)
    print()
    train_size = int(len(small_data) * 0.9)
    print("new train size: ", train_size)
    with open('../data/train.json', 'w+') as f:
        json.dump(small_data[:train_size], f, indent=4)
    with open('../data/val.json', 'w+') as f:
        json.dump(small_data[train_size:], f)
    
    return len(class_map)


if __name__ == "__main__":
    exp_dict = {}
    get_labels('../data/IEMOCAP_full_release/Session1/dialog/EmoEvaluation', exp_dict)
    get_labels('../data/IEMOCAP_full_release/Session2/dialog/EmoEvaluation', exp_dict)
    get_wav_feature('../data/IEMOCAP_full_release/Session1/sentences/wav', exp_dict)
    get_wav_feature('../data/IEMOCAP_full_release/Session2/sentences/wav', exp_dict)
    get_text('../data/IEMOCAP_full_release/Session1/dialog/transcriptions', exp_dict)
    get_text('../data/IEMOCAP_full_release/Session2/dialog/transcriptions', exp_dict)

    #with open('examples.pkl', 'wb') as f:
    #    pickle.dump(exp_dict, f)
    exps = []
    for k, v in exp_dict.items():
        exp_dict[k]['id'] = k
        exps.append(exp_dict[k])

    with open('../data/data.json', 'w') as f:
        json.dump(exps, f, indent=4)

