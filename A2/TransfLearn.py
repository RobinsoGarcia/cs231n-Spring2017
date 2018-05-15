import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import timeit
from PIL import Image
import os
#import matplotlib.pyplot as plt
#%matplotlib inline
hmd = '/home/alien/Documents/CS231n-ConvolutionalNeuralNetworks/A2'
gpu_dtype = torch.cuda.FloatTensor

'''to separate VAL,TRAIN while sampling'''
class ChunkSampler(sampler.Sampler):
     """Samples elements sequentially from some offset.
     Arguments:
         num_samples: # of desired datapoints
         start: offset where we should start selecting from
     """
     def __init__(self, num_samples, start = 0):
         self.num_samples = num_samples
         self.start = start

     def __iter__(self):
         return iter(range(self.start, self.start + self.num_samples))

     def __len__(self):
         return self.num_samples

'''The goal is to perform feature extraction before we start trainning'''
'''DsetTransferL class used a pre-trainned model to extract features '''
'''until the modle's last_layer (index). The extracted feature shall be'''
'''used to train a neural network classifier'''

class DsetTransferL(dset.CIFAR10):
    def __init__(self,**kwargs):

        self.model = kwargs['model']
        self.last_layer = kwargs['last_layer']
        del kwargs['model']
        del kwargs['last_layer']

        if kwargs['train']==True:
            self.file = hmd+'/train_features_dataset.npy'
        else:
            self.file = hmd+'/test_features_dataset.npy'
        super().__init__(**kwargs)

        if self.file in os.listdir():
            print("Features already extracted")
            if kwargs['train']==True:
                self.train_data = np.load(self.file)
            else:
                self.test_data = np.load(self.file)
        else:
            try:
                self.backup_train_data = self.train_data
            except:
                self.backup_train_data = self.test_data

            self.extract()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        return img, target

    def extract(self):
        '''Executes transformation on the entire dataset
        and saves extracted features as train_data'''
        print("starting extraction...")
        model = self.model
        last_layer = self.last_layer
        try:
            data = self.train_data
        except:
            data = self.test_data

        self.num_ftrs = list(model.children())[-last_layer].in_features
        for t,child in enumerate(model.children()):
            child.requires_grad = False
            for tt,param in enumerate(child.parameters()):
                param.requires_grad = False

        model_ext = nn.Sequential(*list(model.children())[:-last_layer]).type(gpu_dtype)

        f = lambda x: model_ext(Variable(x.unsqueeze(0).type(gpu_dtype),requires_grad=False,volatile=True)).view(-1).data
        TT = T.Compose([T.Resize(256),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),T.Lambda(f)])

        new_data = []
        count=0
        for i in data:
            img = Image.fromarray(i)
            new_data.append(TT(img).cpu().numpy())
            if (count%int(len(data)/100))==0:
                print("Images converted: ",count,"/",len(data))
            count+=1

        try:
            self.train_data = np.array(new_data)
        except:
            self.test_data = np.array(new_data)

        np.save(self.file,new_data)

def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train(model,loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):

            x_var = Variable(x.type(gpu_dtype),requires_grad=True)
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var).type(gpu_dtype)

            loss = loss_fn(scores, y_var)
            if (t + 1) % 10 == 0:
                val_acc = check_accuracy(model, loader_val)
                train_acc = check_accuracy(model, loader_train)
                print('t = %d, loss = %.4f, val_acc = %.4f, train_acc = %.4f' % (t + 1, loss.data[0],val_acc,train_acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    #if loader.dataset.train:
    #    print('Checking accuracy on validation set')
    #else:
    #    print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype),requires_grad=False)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

if __name__=="__main__":

    b_size = 128
    NUM_TRAIN = 40000
    NUM_VAL = 10000
    cifar10_train = DsetTransferL(last_layer=1,model=models.resnet50(pretrained=True),root='./cs231n/datasets', train=True, download=True)

    loader_train = DataLoader(cifar10_train, batch_size=b_size, sampler=ChunkSampler(NUM_TRAIN, 0))

    cifar10_val = DsetTransferL(last_layer=1,model=models.resnet50(pretrained=True),root='./cs231n/datasets', train=True, download=True)

    loader_val = DataLoader(cifar10_val, batch_size=b_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    cifar10_test = DsetTransferL(last_layer=1,model=models.resnet50(pretrained=True),root='./cs231n/datasets', train=False, download=True)

    loader_test = DataLoader(cifar10_test, batch_size=64)

    model=models.resnet50(pretrained=True)
    num_ftrs = list(model.children())[-1].in_features
    model = nn.Sequential(nn.BatchNorm1d(num_ftrs),nn.Linear(num_ftrs,num_ftrs),nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),nn.Linear(num_ftrs,10))

    model.apply(reset)
    model.cuda()
    model.type(gpu_dtype)
    torch.cuda.synchronize()

    loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-2)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3,weight_decay=1e-2)

    train(model, loss_fn, optimizer, num_epochs=5)

    check_accuracy(model, loader_test)
