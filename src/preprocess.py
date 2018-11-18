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
    

def get_wav_feature(file_dir, wav_dict):
    for dir_name in os.listdir(file_dir):
        sec_dir = os.path.join(file_dir, dir_name)
        print(sec_dir)
        for file_name in os.listdir(sec_dir):
            file_path = os.path.join(sec_dir, file_name)
            y, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            name = file_name.split('.')[0]
            wav_dict[name]["feat"] = mfcc
            

if __name__ == "__main__":
    exp_dict = {}
    get_labels('../data/IEMOCAP_full_release/Session1/dialog/EmoEvaluation', exp_dict)
    get_labels('../data/IEMOCAP_full_release/Session2/dialog/EmoEvaluation', exp_dict)
    get_wav_feature('../data/IEMOCAP_full_release/Session1/sentences/wav', exp_dict)
    get_wav_feature('../data/IEMOCAP_full_release/Session2/sentences/wav', exp_dict)
    with open('examples.pkl', 'wb') as f:
        pickle.dump(exp_dict, f)
