from flask import Flask, request
from simple_model import SER
import librosa
import torch as th
import pandas as pd
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO

app = Flask(__name__)

@app.route('/')
def hello():
    return "This is a server"

@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json(force=True)
    feature_type = data['feat_type']
    assert feature_type in ['raw', 'mfcc', 'all']
    model_input = data['m_in']

    y = pd.read_json(data['feat']).as_matrix().squeeze()
    # get the sampled raw signal
    if feature_type == 'raw':
        sr = data['sr']
        if model_input == 'mfcc':
            feat = librosa.feature.mfcc(y=y, sr=int(sr), n_mfcc=40).T # (S, 40)
            feat = th.FloatTensor(feat).unsqueeze(0)
        elif model_input == 'all':
            all_feats, f_names = audioFeatureExtraction.stFeatureExtraction(y, sr, 2048, 512)
            feat = th.FloatTensor(all_feats.T).unsqueeze(0)
             
    # get the mfcc directly
    elif feature_type == 'mfcc': 
        feat = th.FloatTensor(y).view(-1, 40).unsqueeze(0)

    elif feature_type == 'all':
        feat = th.FloatTensor(y).view(-1, 34).unsqueeze(0)
    

    # load model
    model = SER(h_size=200, feat_size=feat.size(2), class_num=4, dropout=0.)
    #model.cuda()
    model.eval()
    #model.load_state_dict(th.load('checkpoint/model.pt'))
    pred = model(feat, [feat.size(1)], None, None)
    pred = pred.max(dim=1)[1].item()
    
    return str(pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=101)
    
