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
import foolbox as fb
import foolbox.attacks as fa

# Use CUDA
use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(3)

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

save_path = "./save/adv_train"

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

def attack(inputs, soft_labels, model):
    losses = AverageMeter()
    accs = AverageMeter()
    # model.eval()
    # switch to train mode
    # model.train()
    epsilons = [
        # 0.001,
        # 0.0015,
        # 0.002,
        # 0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.1,
        0.3,
        0.5,
    ]

    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        # fa.LinfBasicIterativeAttack(),
        # fa.LinfAdditiveUniformNoiseAttack(),
        # fa.LinfDeepFoolAttack(),     
    ]

    attack = attacks[random.randint(0,len(attacks)-1)]
    raw, clipped, is_adv = attack(model, inputs, soft_labels.argmax(dim=1), epsilons=epsilons[random.randint(0,len(epsilons)-1)]) 

    return clipped

def main():

    for arch in ['resnet50', 'densenet121']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
        # Model

        model = load_model(arch)
        # model.load_state_dict(torch.load("commit/baseline/{}.pth.tar".format(arch))['state_dict'])
        # transform model bounds
        
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(args['epochs'])):

            train_loss, train_acc = train(trainloader, model, optimizer)
            print(args)
            print('acc: {}, loss:{}'.format(train_acc, train_loss))
            
            with open( os.path.join(save_path, "log.txt"),'a' ) as f:
                f.write('arch: {}, epoch: {}, acc: {}, loss: {}\n'.format(arch,epoch,train_acc,train_loss))

            # save model
            best_acc = max(train_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()

        print('Best acc:')
        print(best_acc)



def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()    # switch to train mode
    model.train()
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, preprocessing=preprocessing, bounds=bounds)

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        inputs = attack(inputs, soft_labels, fmodel)
        
        for input in inputs:
            # print(input.shape)
            input = transform(input)
        
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
