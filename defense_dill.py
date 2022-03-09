from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy

import argparse

# Use CUDA
use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(3)

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

save_path = "save/all_data/"


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load(os.path.join(save_path,'data.npy'))
        labels = np.load(os.path.join(save_path,'label.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        # assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def eval(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    # model.train()
    
    soft_label = []

    for (inputs, soft_labels) in tqdm(trainloader):
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        # inputs.requires_grad = True
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        # optimizer.zero_grad()
        # model.zero_grad()
        # loss.backward() 
        
        outputs = model(inputs)
        acc = accuracy(outputs, targets)
        outputs = F.softmax(outputs/3,dim=1)
        soft_label.append(outputs.cpu().detach().numpy())        
        
        # optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
        
        # print(attack_image.shape)

    soft_label = np.concatenate(soft_label)
    return losses.avg, accs.avg, soft_label

def main():

    for arch in ['densenet121']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        # Data
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
        # Model

        model = load_model(arch)
        model.load_state_dict(torch.load("save/all_data/{}.pth.tar".format(arch))['state_dict'])
        
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(args['epochs'])):

            loss, acc, soft_labels = eval(trainloader, model, optimizer)
            print(args)
            print('acc: {}'.format(acc))
            print(soft_labels.shape)
            
            np.save("save/all_data/soft_label.npy", soft_labels)
            print(soft_labels)
            
            # with open( os.path.join(save_path, "log.txt"),'a' ) as f:
            #     f.write('arch: {}, epoch: {}, acc: {}\n'.format(arch,epoch,train_acc))

            # # save model
            # best_acc = max(train_acc, best_acc)
            # save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'acc': train_acc,
            #         'best_acc': best_acc,
            #         'optimizer' : optimizer.state_dict(),
            #     }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()
            break
            
        # print('Best acc:')
        # print(best_acc)


def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def save_checkpoint(state, arch):
    filepath = os.path.join(save_path, arch + '.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    main()
