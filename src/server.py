from flask import Flask, request
from simple_model import SER
import librosa
import torch as th
import pandas as pd
app = Flask(__name__)
@app.route('/')
def hello():
    return "This is a server"

@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json(force=True)
    y = pd.read_json(data['mfcc']).as_matrix().squeeze()
    print(type(y))
    sr = data['sr']
    mfcc = librosa.feature.mfcc(y=y, sr=int(sr), n_mfcc=40).T # (S, 40)
    mfcc = th.FloatTensor(mfcc).unsqueeze(0)
    # load model
    model = SER(h_size=200, feat_size=mfcc.size(2), class_num=4, dropout=0.)
    model.cuda()
    model.eval()
    model.load_state_dict(th.load('checkpoint/model.pt'))
    pred = model(mfcc, [mfcc.size(1)], None, None)
    
    return pred

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=101)
    
