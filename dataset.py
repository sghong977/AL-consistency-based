import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import glob

def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()
    elif name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()
    elif name == 'CIFAR100':
        return get_CIFAR100()
    elif name == 'STL10':
        return get_STL10()
    elif name == 'ImageNet':
        return get_ImageNet()

def get_ImageNet():
    lb = [0,2,3,4,5,6]
    dset = datasets.ImageFolder(root='../../../nas/Public/imagenet/train')
    #img_folder_val = datasets.ImageFolder(root='../../../nas/Public/imagenet/val')

    #data_tr = img_folder_train
    #data_te = img_folder_val
#    data_tr = datasets.ImageNet('./ImageNet', split='train', download=True)
#    data_te = datasets.ImageNet('./ImageNet', split='val', download=True)
    #=X_tr = data_tr #.train_data
    #Y_tr = torch.from_numpy(np.array(train_labels))   #data_tr.label
    #X_te = data_te.test_data
    #Y_te = torch.from_numpy(np.array(data_te.test_labels))
    #print("get data fin")
    X_tr = DataLoader(dset[[lb]]) #.train_data
    print(type(X_tr), len(X_tr))

    """
    X_tr = list()
    pth = '../../../nas/Public/imagenet/train/'
    aa = os.listdir(pth)
    print(aa[0])
    for a in aa:
        for f in glob.glob(pth + a + "/*"):
            im = Image.open(f)
            X_tr.append(im)
    print(np.array(X_tr).shape)
    ss"""
    return X_tr, Y_tr, X_te, Y_te

def get_STL10():
    data_tr = datasets.STL10('./STL10', split='train', download=True)
    data_te = datasets.STL10('./STL10', split='test', download=True)
    X_tr = torch.from_numpy(data_tr.data)
    Y_tr = torch.LongTensor(data_tr.labels)
    X_te = torch.from_numpy(data_te.data)
    Y_te = torch.LongTensor(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100():
    data_tr = datasets.CIFAR100('./CIFAR100', train=True, download=True)
    data_te = datasets.CIFAR100('./CIFAR100', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('./FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST('./FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN():
    data_tr = datasets.SVHN('./SVHN', split='train', download=True)
    data_te = datasets.SVHN('./SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10():
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR100':
        return DataHandler3
    elif name == 'STL10':
        return DataHandler4


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

# for STL10
class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.numpy()
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
