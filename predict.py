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

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

save_path = "save/adv_train_pgd_inf0.1_loadbaseline"


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        # images = np.load(os.path.join(save_path,'data.npy'))
        # labels = np.load(os.path.join(save_path,'label.npy'))
        images = np.load(os.path.join(save_path, "attack_data_densenet121.npy"))
        labels = np.load(os.path.join(save_path, "attack_label_densenet121.npy"))
        
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

def main():

    for arch in ['resnet50']:
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
        trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        # Model

        model = load_model(arch)        
        model.load_state_dict(torch.load("save/adv_train_pgd_inf0.1_loadbaseline/resnet50.pth.tar")['state_dict']) 
        
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in range(1):

            train_loss, train_acc, classes = data_clean(trainloader, model, optimizer)
            print(args)
            print('acc: {}'.format(train_acc))
            # print('loss: {}'.format(train_loss))



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

def data_clean(trainloader, model, optimizer):
    losses = []
    accs = []
    classes = []
    for (inputs, soft_labels) in tqdm(trainloader):
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        print(F.softmax(outputs))
        loss = cross_entropy(outputs, soft_labels)
        # print(loss.cpu().item())
        losses.append((loss.cpu().item()))
        acc = accuracy(outputs, targets)
        # print(acc)
        accs.append(acc[0].cpu().item())
        # print(soft_labels,outputs)
        _, c = torch.max(outputs.cpu(), dim=1)
        classes.append(c)
        # print(c)
        # exit()
        # print( torch.max(outputs, dim=1))
        # outputs = outputs[0].cpu().item()
        # print(outputs)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # losses.update(loss.item(), inputs.size(0))
        # accs.update(acc[0].item(), inputs.size(0))
        # break
        
    return losses, accs, classes

def write_excel(losses, accs, classes):
    labels = np.load("label_all.npy")
    labels = np.argmax(labels,axis=1)
    label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk']
    # import xlwt
    import xlsxwriter as xlwt
    workbook = xlwt.Workbook("./data-0.xlsx")  # 新建一个工作簿
    sheet = workbook.add_worksheet("cifar-10")  # 在工作簿中新建一个表格
    
    sheet.write(0,0, "index")
    sheet.write(0,1, "image")
    sheet.write(0,2, "label")
    sheet.write(0,3, "predict result")
    sheet.write(0,4, "acc")
    sheet.write(0,5, "loss")
    
    sheet.set_column('B:B', 32)
      
    for i in range(60000):
        if i % 10000 == 0 and i !=0 :
            workbook.close()
            workbook = xlwt.Workbook("./data-{}.xlsx".format(i//10000))  # 新建一个工作簿
            sheet = workbook.add_worksheet("cifar-10")  # 在工作簿中新建一个表格
            sheet.write(0,0, "index")
            sheet.write(0,1, "image")
            sheet.write(0,2, "label")
            sheet.write(0,3, "predict result")
            sheet.write(0,4, "acc")
            sheet.write(0,5, "loss")
            sheet.set_column('B:B', 32)
            
        image = "data_vis/images/{}.jpg".format(i)
        acc = accs[i]
        loss = losses[i]
        label = labels[i]
        sheet.set_row(i, 32)   
            
        sheet.write(i%10000+1, 0, i)  # 像表格中写入数据（对应的行和列）
        sheet.insert_image(i%10000+1, 1, image)
        sheet.write(i%10000+1, 2, label_list[label])
        sheet.write(i%10000+1, 3, label_list[classes[i]])
        sheet.write(i%10000+1, 4, acc)
        sheet.write(i%10000+1, 5, loss)
        # workbook.save()  # 保存工作簿
        # break
        
            
    workbook.close()

def save_checkpoint(state, arch):
    filepath = os.path.join(save_path, arch + '.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    main()
