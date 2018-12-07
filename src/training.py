import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from dataloader import *
from breath_dataloader import *
from simple_model import SER, SER_CNN, SER_RNN_Encoder
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
    patience = 40
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
        w = th.FloatTensor([0.28, 0.19, 0.28, 0.18, 0.07]).to(device) 
        loss_fn = nn.CrossEntropyLoss(weight=w)
        print('weighted')
    else:
        loss_fn = nn.CrossEntropyLoss()
        print('unweighted')

    optim = Adam(model.parameters(), lr=5e-4)
    epoch = 100
    max_acc = 0
    patience = 20
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
    th.save(best_model.state_dict(), 'checkpoint/model.pt')

    return best_model


def evaluation(model, loader, device):
    idx2emo = {0: 'ang', 1: 'hap', 2: 'sad', 3: 'neu'}
    weight = [1103, 1636, 1084, 1708]
    weight = [x / np.sum(weight) for x in weight]

    with th.no_grad():
        model.eval()
        total = 0
        correct = 0
        class_correct_count = [0] * len(idx2emo)
        class_total_count = [0] * len(idx2emo)
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
            # get each class acc
            for pclass, tclass in zip(c,labels):
                class_correct_count[tclass] += 1 if pclass == tclass else 0
                class_total_count[tclass] += 1
            
    
    WA = 0
    for nc in range(len(idx2emo)):
        WA += (class_correct_count[nc] / class_total_count[nc]) * weight[nc]
        print("class: {}, accuracy:{:.3f}".format(idx2emo[nc], class_correct_count[nc] / class_total_count[nc]))
    print("unweighted accuracy:{:.3f}".format(correct / total))
    print("weighted accuracy:{:.3f}".format(WA))
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

    print("Feat: %s" % args.feat)
    # padding type for SER
    if args.speech_model == 'SER':
        pad_type = 'batch'
    elif args.speech_model[:-2] == 'SER_CNN':
        pad_type = 'global'
    # process the data
    class_num = recategorize_and_split(args.data_path)
    trainloader = get_dataloader(args.train_path, batch_size=16, shuffle=True, feat_size=args.feat_size, pad_type=pad_type)
    valloader = get_dataloader(args.val_path, batch_size=2, shuffle=False, feat_size=args.feat_size, pad_type=pad_type)
    if args.feat == "speech":
        print("Model: %s" % args.speech_model)
        actual_feat = 21
        if args.speech_model == 'SER':
            model = SER(h_size=200, feat_size=actual_feat, class_num=class_num, dropout=0.)
        elif args.speech_model in ['SER_CNN1d', 'SER_CNN2d']:
            conv_type = args.speech_model[-2:]
            model = SER_CNN(conv_type=conv_type, h_size=100, feat_size=actual_feat, class_num=class_num, \
                    max_time_step=trainloader.dataset.max_time_step, nlayers=2, kernel=[3], dropout=0.3)
        model.to(device)

    # breathing model
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
        print("Model: %s" % args.model)
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
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default="../data/data.pkl")
    parser.add_argument('-train_path', type=str, default="../data/train.pkl")
    parser.add_argument('-val_path', type=str, default="../data/val.pkl")
    parser.add_argument('-feat', type=str, default="speech", choices=['speech', 'text', 'mix', 'breath'])
    parser.add_argument('-model', type=str, default="TER_RNN", choices=['TER_RNN', 'TER_FFNN'])
    parser.add_argument('-speech_model', type=str, default="SER", choices=['SER', 'SER_CNN1d', 'SER_CNN2d'])
    parser.add_argument('-embs_size', type=int, default=100, choices=[50, 100, 200, 30])
    parser.add_argument('-feat_size', type=int, default=40, help='the size of acoustic feature')
    parser.add_argument('-pretrain_embs', action='store_true', default=False, help='whether to use pretrain embeddings')
    parser.add_argument('--cuda', type=int, default=0, help='the cuda device to use')
    parser.add_argument('--weighted', action='store_true', default=False, help='the cuda device to use')
    parser.add_argument('-description', type=str, default='', help='the model config')
    args = parser.parse_args()
    main(args)











