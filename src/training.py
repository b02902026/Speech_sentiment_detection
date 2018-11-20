import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from dataloader import *
from simple_model import SER
from preprocess import recategorize_and_split
import argparse
import time

def training(trainloader, valloader, class_num):
    
    loss_fn = nn.CrossEntropyLoss()
    model = SER(h_size=200, feat_size=40, class_num=class_num)
    optim = Adam(model.parameters(), lr = 1e-4)
    epoch = 100
    for e in range(epoch):
        total_loss = 0
        start_time = time.time()
        for i, (feat, length, labels, text) in enumerate(trainloader):
            optim.zero_grad()
            pred = model(feat, length) 
            loss = loss_fn(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()
            print("epoch:{}/{}, batch:{}/{}, loss:{}".format(e, epoch, i, len(trainloader), total_loss / (i+1)))
        
        print("epoch {} finished, elapsed time {} sec".format(e, time.time()-start_time))
        validation(model, valloader, loss_fn)
        evaluation(model, valloader)
    
    return model

def evaluation(model, loader):
    model.eval()
    total = 0
    correct = 0
    for i, (feat, length, labels, _) in enumerate(loader):
        B = feat.size(0)
        pred = model(feat, length) 
        _, c = pred.max(dim=1)
        correct += th.sum(c == labels).item()
        total += B
    
    print("acc:{}".format(correct / total))
    model.train()
    

def validation(model, valloader, loss_fn):
    model.eval()
    total_loss = 0
    for i, (feat, length, labels, _) in enumerate(valloader):
        pred = model(feat, length) 
        loss = loss_fn(pred, labels)
        total_loss += loss.data.item()

    print("validation loss:{}".format(total_loss / len(valloader)))
    model.train()

def main(args):
    class_num = recategorize_and_split(args.data_path)
    trainloader = get_dataloader(args.train_path, batch_size=16, shuffle=False)
    valloader = get_dataloader(args.val_path, batch_size=2, shuffle=False)
    training(trainloader, valloader, class_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default="../data/data.json")
    parser.add_argument('-train_path', type=str, default="../data/train.json")
    parser.add_argument('-val_path', type=str, default="../data/val.json")
    args = parser.parse_args()
    main(args)











