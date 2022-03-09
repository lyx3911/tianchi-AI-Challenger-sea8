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

save_path = "./"

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

def attack(trainloader, model, optimizer):
    # model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    # model.eval()
    # switch to train mode
    # model.train()
    # epsilons = [
    #     # 0.001,
    #     # 0.0015,
    #     # 0.002,
    #     # 0.003,
    #     0.005,
    #     0.01,
    #     0.02,
    #     0.03,
    #     0.1,
    #     0.3,
    #     0.5,
    # ]

    # attacks = [
    #     fa.FGSM(),
    #     fa.LinfPGD(),
    #     fa.L2PGD()
    #     # fa.LinfBasicIterativeAttack(),
    #     # fa.LinfAdditiveUniformNoiseAttack(),
    #     # fa.LinfDeepFoolAttack(),
        
    #     # # 2021-12-16 加的 foolbox2
    #     # fa.L2PGD(), 
    #     # fa.SaltAndPepperNoiseAttack(),
    #     # fa.L2DeepFoolAttack(),
    #     # fa.VirtualAdversarialAttack(steps=10),
    #     # fa.L2ProjectedGradientDescentAttack(),      
    # ]
    attacks = [fa.L2PGD(), fa.FGM()]
    epsilons = [2.5, 2.5]
    attack_image = []
    attack_label = []

    for (inputs, soft_labels) in tqdm(trainloader):
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        
        for i in range(2):
            attack = attacks[i]
            raw, clipped, is_adv = attack(model, inputs, soft_labels.argmax(dim=1), epsilons=epsilons[i]) 
            # sample_epsilons = random.sample(epsilons,4)
            # for epsilon in sample_epsilons:
                # raw, clipped, is_adv = attack(model, inputs, soft_labels.argmax(dim=1), epsilons=epsilon)
            # print(raw[is_adv==True].shape)
            # attack_image.append(raw[is_adv==True].cpu().detach().numpy().transpose(0, 2, 3, 1))
            # attack_label.append(soft_labels[is_adv==True].cpu().detach().numpy())  
            attack_image.append(clipped[is_adv==True].cpu().detach().numpy().transpose(0, 2, 3, 1))
            attack_label.append(soft_labels[is_adv==True].cpu().detach().numpy())  
            # print("attack 1")              
        # break
        # print(attack_image.shape)
    attack_image = np.concatenate(attack_image)
    attack_label = np.concatenate(attack_label)
    print(attack_image.shape, attack_label.shape)
    return attack_image, attack_label

def main():

    for arch in ['resnet50', 'densenet121']:
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
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
        # Model

        model = load_model(arch)
        model.load_state_dict(torch.load("commit/baseline-lr0.01/{}.pth.tar".format(arch))['state_dict'])
        # transform model bounds
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        bounds = (0, 1)
        fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
        
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(args['epochs'])):

            attack_data, attack_label = attack(trainloader, fmodel, optimizer)
            # print(args)
            # print('acc: {}'.format(attack_acc))
            
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
            
            # invert transpose (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # attack_data[:,:,:,0] = attack_data[:,:,:,0]*0.2023+0.4914
            # attack_data[:,:,:,1] = attack_data[:,:,:,1]*0.1994+0.4822
            # attack_data[:,:,:,2] = attack_data[:,:,:,2]*0.2010+0.4465
            attack_data = attack_data*255
            attack_data = np.clip(attack_data, a_min=0, a_max=255)
            attack_data = attack_data.astype(np.uint8)
            
            np.save("./attack_data/foolbox-attack-baselinelr0.01-L2_data_{}.npy".format(arch), attack_data)
            np.save("./attack_data/foolbox-attack-baselinelr0.01-L2_label_{}.npy".format(arch), attack_label)
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
