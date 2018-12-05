import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from dataloader import *
from breath_dataloader import *
from simple_model import SER, SER_CNN
from text_model import TER_FFNN, TER_RNN
from mix_model import MixER
from breath_model import BreathClassifier
from preprocess import recategorize_and_split, Vocabulary
import argparse
import time
import pickle, json
from copy import deepcopy

CUDA = th.cuda.is_available()


def breath_training(trainloader, valloader, class_num, device):
    loss_fn = nn.CrossEntropyLoss()
    model = BreathClassifier(h_size=200, kernel_size=[2,3,4,5,6], class_num=class_num, mix=True, ser_feat=54)
    model.to(device)
    #model = SER(h_size=200, feat_size=54, class_num=class_num, dropout=0.2)
    #model.to(device)
    optim = Adam(model.parameters(), lr = 5e-4)
    
    epoch = 100
    max_acc = 0.
    max_valloss = float('Inf')
    patience = 30
    k = patience
    best_model = None

    for e in range(epoch):
        total_loss = 0
        start_time = time.time()
        for i, (feat, length, labels, tokens_id, tok_length, breath, breath_length) in enumerate(trainloader):
            optim.zero_grad()
            feat = feat.to(device)
            labels = labels.to(device)
            breath = breath.to(device)
            th.cuda.empty_cache()
            pred = model(breath, feat, length) 
            #pred = model(feat, length, None, None)
            loss = loss_fn(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()
            #print("\repoch:{}/{}, batch:{}/{}, loss:{}".format(e, epoch, i, len(trainloader), total_loss / (i+1)), end='')
        
        print("\nepoch {} finished, elapsed time {} sec".format(e, time.time()-start_time))
        valloss = breath_validation(model, valloader, loss_fn, device)
        acc = breath_evaluation(model, valloader, device)
        if acc >= max_acc:
            k = patience
            max_acc = acc
            best_model = deepcopy(model)
        else:
            k -= 1

        if k == 0:
            break

    print("{}Early Stopping{}".format('-'*30, '-'*30))
    breath_validation(best_model, valloader, loss_fn, device)
    breath_evaluation(best_model, valloader, device)

    return best_model

def breath_evaluation(model, loader, device):
    with th.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, (feat, length, labels, tokens_id, tok_length, breath, _) in enumerate(loader):
            B = feat.size(0)
            feat = feat.to(device)
            breath = breath.to(device)
            labels = labels.to(device)
            tokens_id = tokens_id.to(device)
            th.cuda.empty_cache()
            pred = model(breath, feat, length) 
            #pred = model(feat, length, None, None) 
            _, c = pred.max(dim=1)
            correct += th.sum(c == labels).item()
            total += B
    
    print("acc:{}".format(correct / total))
    model.train()
    return correct / total

def breath_validation(model, valloader, loss_fn, device):
    model.eval()
    total_loss = 0
    for i, (feat, length, labels, _, _, breath, _) in enumerate(valloader):
        feat = feat.to(device)
        breath = breath.to(device)
        labels = labels.to(device)
        th.cuda.empty_cache()
        pred = model(breath, feat, length) 
        #pred = model(feat, length, None, None) 
        loss = loss_fn(pred, labels)
        total_loss += loss.data.item()

    print("validation loss:{}".format(total_loss / len(valloader)))
    model.train()

def training(model ,trainloader, valloader, class_num, device, args):
    if args.weighted:
        weight = th.FloatTensor([0.28, 0.19, 0.28, 0.18, 0.07]).to(device) 
        print('weighted')
    else:
        weight = None
        print('unweighted')

    loss_fn = nn.CrossEntropyLoss(weight=weight)
    #model = SER(h_size=200, feat_size=40, class_num=class_num)
    #model.to(device)
    
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    epoch = 100
    max_acc = 0
    patience = 25
    k = patience
    best_model = None
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
            if i % 100 == 0:
                print("epoch:{}/{}, batch:{}/{}, loss:{}".format(e, epoch, i, len(trainloader), total_loss / (i+1)))
        
        print("epoch {} finished, elapsed time {} sec".format(e, time.time()-start_time))
        validation(model, valloader, loss_fn, device)
        acc = evaluation(model, valloader, device)
        if acc > max_acc:
            max_acc = acc
            best_model = deepcopy(model)
            k = patience
        else:
            k -= 1
        if k == 0:
            break

    print("{}Early Stopping{}".format('-'*30, '-'*30))
    validation(best_model, valloader, loss_fn, device)
    evaluation(best_model, valloader, device)
    
    return best_model


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
        model = SER(h_size=200, feat_size=args.feat_size, class_num=class_num, dropout=0.)
        #model = SER_CNN(conv_type='2d', h_size=100, feat_size=40, class_num=class_num, max_time_step=trainloader.dataset.max_time_step, nlayers=2, kernel=[3], dropout=0.5)
        model.to(device)
    elif args.feat == "breath":
        class_num = recategorize_and_split(args.data_path)
        trainloader = get_breath_dataloader(args.train_path, batch_size=2, shuffle=True)
        valloader = get_breath_dataloader(args.val_path, batch_size=2, shuffle=False)
        breath_training(trainloader, valloader, class_num, device)
        exit()


    elif args.feat == "text":
        if args.pretrain_embs:
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
                            'nlayers': 2,
                            'dropout': 0.2,
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
        model_kwargs = {'speech_feat_size': args.feat_size,
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

    training(model, trainloader, valloader, class_num, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default="../data/data.pkl")
    parser.add_argument('-train_path', type=str, default="../data/train.pkl")
    parser.add_argument('-val_path', type=str, default="../data/val.pkl")
    parser.add_argument('-feat', type=str, default="speech", choices=['speech', 'text', 'mix', 'breath'])
    parser.add_argument('-model', type=str, default="TER_RNN", choices=['TER_RNN', 'TER_FFNN'])
    parser.add_argument('-embs_size', type=int, default=100, choices=[50, 100, 200, 30])
    parser.add_argument('-feat_size', type=int, default=40, help='the size of acoustic feature')
    parser.add_argument('-pretrain_embs', action='store_true', default=False, help='whether to use pretrain embeddings')
    parser.add_argument('--cuda', type=int, default=0, help='the cuda device to use')
    parser.add_argument('--weighted', action='store_true', default=False, help='the cuda device to use')
    args = parser.parse_args()
    main(args)











