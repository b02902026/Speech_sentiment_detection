import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from dataloader import *
from simple_model import SER, SER_CNN
from text_model import TER_FFNN, TER_RNN
from mix_model import MixER
from preprocess import recategorize_and_split, Vocabulary
import argparse
import time
import pickle, json


CUDA = th.cuda.is_available()


def training(model, trainloader, valloader, class_num, device):
    
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    epoch = 100
    max_acc = 0
    for e in range(epoch):
        total_loss = 0
        start_time = time.time()
        for i, (feat, length, labels, tokens_id, tok_length) in enumerate(trainloader):
            optim.zero_grad()
            feat = feat.to(device)
            labels = labels.to(device)
            tokens_id = tokens_id.to(device)
            th.cuda.empty_cache()
            pred = model(feat, length, tokens_id, tok_length)
            loss = loss_fn(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()
            if (i+1) % 100 == 0:
                print("epoch:{}/{}, batch:{}/{}, loss:{}".format(e, epoch, i, len(trainloader), total_loss / (i+1)))
        
        validation(model, valloader, loss_fn, device)
        acc = evaluation(model, valloader, device)
        if acc > max_acc:
            max_acc = acc
        print('max acc: {}'.format(max_acc))
        print("epoch {} finished, elapsed time {} sec".format(e, time.time()-start_time))
        print()
    
    return model

def evaluation(model, loader, device):
    with th.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, (feat, length, labels, tokens_id, tok_length) in enumerate(loader):
            B = feat.size(0)
            feat = feat.to(device)
            labels = labels.to(device)
            tokens_id = tokens_id.to(device)
            th.cuda.empty_cache()
            pred = model(feat, length, tokens_id, tok_length)
            _, c = pred.max(dim=1)
            correct += th.sum(c == labels).item()
            total += B
    
    print("acc:{}".format(correct / total))
    model.train()
    return correct / total
    

def validation(model, valloader, loss_fn, device):
    with th.no_grad():
        model.eval()
        total_loss = 0
        for i, (feat, length, labels, tokens_id, tok_length) in enumerate(valloader):
            feat = feat.to(device)
            labels = labels.to(device)
            tokens_id = tokens_id.to(device)
            th.cuda.empty_cache()
            pred = model(feat, length, tokens_id, tok_length)
            loss = loss_fn(pred, labels)
            total_loss += loss.data.item()

    print("validation loss:{}".format(total_loss / len(valloader)))
    model.train()


def main(args):
    if CUDA:
        device = th.device("cuda:{}".format(args.cuda))
        print("CUDA Enabled")
    else:
        device = th.device("cpu")
        print("CUDA Disabled")
    class_num = recategorize_and_split(args.data_path)
    trainloader = get_dataloader(args.train_path, batch_size=16, shuffle=True)
    valloader = get_dataloader(args.val_path, batch_size=2, shuffle=False)
    if args.feat == "speech":
        model = SER(h_size=200, feat_size=40, class_num=class_num, dropout=0.)
        #model = SER_CNN(conv_type='2d', h_size=100, feat_size=40, class_num=class_num, max_time_step=trainloader.dataset.max_time_step, nlayers=2, kernel=[3], dropout=0.5)
        model.to(device)
    elif args.feat == "text":
        with open('../data/glove.6B.%dd-relativized.pkl' % args.embs_size, 'rb') as f:
            pretrain_embs = pickle.load(f)
        with open('../data/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        if args.model == 'TER_FFNN':
            model_kwargs = {'ntoken': len(vocab),
                            'emb_size': args.embs_size,
                            'nhid': 200,
                            'class_num': class_num,
                            'nlayers': 1,
                            'dropout': 0.,
                            'embs_dropout': 0.2,
                            'embs_fixed': False,
                            'pretrain_embs': pretrain_embs if args.pretrain_embs else None}
            model = TER_FFNN(**model_kwargs)
            model.to(device)
        elif args.model == 'TER_RNN':
            model_kwargs = {'cell_type': 'LSTM',
                            'ntoken': len(vocab),
                            'emb_size': args.embs_size,
                            'ninp': args.embs_size,
                            'nhid': 200,
                            'nlayers': 1,
                            'dropout': 0.,
                            'bidirection': True,
                            'class_num': class_num,
                            'fc_nhid': 200,
                            'fc_nlayers': 1,
                            'fc_dropout': 0.,
                            'embs_dropout': 0.2,
                            'embs_fixed': False,
                            'pretrain_embs': pretrain_embs if args.pretrain_embs else None}
            model = TER_RNN(**model_kwargs)
            model.to(device)
    elif args.feat == "mix":
        with open('../data/glove.6B.%dd-relativized.pkl' % args.embs_size, 'rb') as f:
            pretrain_embs = pickle.load(f)
        with open('../data/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        model_kwargs = {'speech_feat_size': 40,
                        'cell_type': 'GRU',
                        'ntoken': len(vocab),
                        'emb_size': args.embs_size,
                        'ninp': args.embs_size,
                        'nhid': 100,
                        'nlayers': 2,
                        'dropout': 0.4,
                        'bidirection': True,
                        'class_num': class_num,
                        'fc_nhid': 200,
                        'fc_nlayers': 2,
                        'fc_dropout': 0.4,
                        'embs_dropout': 0.4,
                        'embs_fixed': False,
                        'pretrain_embs': pretrain_embs if args.pretrain_embs else None}
        model = MixER(**model_kwargs)
        model.to(device)
    training(model, trainloader, valloader, class_num, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('-data_path', type=str, default="../data/data.json")
    #parser.add_argument('-train_path', type=str, default="../data/train.json")
    #parser.add_argument('-val_path', type=str, default="../data/val.json")
    parser.add_argument('-data_path', type=str, default="../data/data.pkl")
    parser.add_argument('-train_path', type=str, default="../data/train.pkl")
    parser.add_argument('-val_path', type=str, default="../data/val.pkl")
    parser.add_argument('-feat', type=str, default="speech", choices=['speech', 'text', 'mix'])
    parser.add_argument('-model', type=str, default="TER_RNN", choices=['TER_RNN', 'TER_FFNN'])
    parser.add_argument('-embs_size', type=int, default=100, choices=[50, 100, 200, 30])
    parser.add_argument('-pretrain_embs', action='store_true', default=False, help='whether to use pretrain embeddings')
    parser.add_argument('--cuda', type=int, default=0, help='the cuda device to use')
    args = parser.parse_args()
    main(args)











