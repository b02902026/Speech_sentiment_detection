import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from dataloader import *
from simple_model import SER
from preprocess import recategorize_and_split
import argparse

def training(trainloader, valloader, class_num):
    
    loss_fn = nn.CrossEntropyLoss()
    model = SER(h_size=128, feat_size=40, class_num=class_num)
    optim = Adam(model.parameters(), lr = 1e-3)
    epoch = 10
    for e in range(epoch):
        total_loss = 0
        for i, (feat, length, labels, text) in enumerate(trainloader):
            optim.zero_grad()
            pred = model(feat, length) 
            loss = loss_fn(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()
            print("epoch:{}/{}, batch:{}/{}, loss:{}".format(e, epoch, i, len(trainloader), total_loss / (i+1)))

        validation(model, valloader)
    
    return model

def validation(modal, valloader):
    model.eval()
    for i, (feat, length, labels) in enumerate(valloader):
        pred = SER(feat, length) 
        loss = loss_fn(pred, labels)
        total_loss += loss.data.item()

    print("validation loss:{}".format(total_loss / len(valloader)))
    model.train()

def main(args):
    class_num = recategorize_and_split(args.data_path)
    trainloader = get_dataloader(args.train_path, batch_size=2, shuffle=True)
    valloader = get_dataloader(args.val_path, batch_size=2, shuffle=False)
    training(trainloader, valloader, class_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default="../data/data.json")
    parser.add_argument('-train_path', type=str, default="../data/train.json")
    parser.add_argument('-val_path', type=str, default="../data/val.json")
    args = parser.parse_args()
    main(args)











