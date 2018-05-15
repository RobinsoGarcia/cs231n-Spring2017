import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import timeit


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


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train(model,model_ext, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x,volatile=True)#.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())
            extracted = model_ext(x_var).view(16,-1).type(gpu_dtype)
            extracted.volatile=False
            scores = model(extracted)
            print(scores.is_cuda)

            loss = loss_fn(scores, y_var.long()) #+ 0.3*(loss_fn(h1, y_var.long()) + loss_fn(h2, y_var.long()))
            if (t + 1) % 10 == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

class Inception(torch.nn.Module):
    def __init__(self, C_in,C_bot,C_out):
        super(Inception,self).__init__()
        self.cuda()
        c11_out,c33_out,c55_out,pool_proj = C_out
        c33_bot,c55_bot = C_bot

        self.conv11 = torch.nn.Conv2d(C_in,c11_out,kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv33 = torch.nn.Conv2d(c33_bot,c33_out,kernel_size=3,padding=1)
        self.conv11_33 = torch.nn.Conv2d(C_in,c33_bot,kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv55 = torch.nn.Conv2d(c55_bot,c55_out,kernel_size=5,padding=2)
        self.conv11_55 = torch.nn.Conv2d(C_in,c55_bot,kernel_size=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.pool = torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv11_pool = torch.nn.Conv2d(C_in,pool_proj,kernel_size=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        #print(x.size())
        c1 = self.relu1(self.conv11(x))
        c2 = self.relu3(self.conv33(self.relu2(self.conv11_33(x))))
        c3 = self.relu5(self.conv55(self.relu4(self.conv11_55(x))))
        c4 = self.relu6(self.conv11_pool(self.pool(x)))

        return torch.cat((c1,c2,c3,c4),1)

class GoogLeNet(torch.nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.cuda()
        self.bn1 = nn.BatchNorm2d(3,eps=1e-5)
        self.conv1 = nn.Conv2d(3,64,kernel_size=6,stride=2)#14
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1)#10
        #self.bn2 = nn.BatchNorm2d(64,eps=1e-5)
        self.conv2 = nn.Conv2d(64,192,kernel_size=5,stride=1)#6
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=4,stride=1)# 3
        #self.bn3 = nn.BatchNorm2d(192,eps=1e-5)###########
        self.incep1 = Inception(192,(96,16),(64,128,32,32))
        self.incep2 = Inception(256,(128,32),(128,192,96,64))
        self.pool3 = nn.MaxPool2d(kernel_size=1,stride=1)#3
        self.incep3 = Inception(480,(96,16),(192,208,48,64))
        size_lin = int(512*(3**2))
        self.lin_buf11 = nn.Linear(size_lin,int(0.5*size_lin))
        self.lin_buf12 = nn.Linear(int(0.5*size_lin),10)
        self.relu3 = nn.ReLU(inplace=True)
        self.incep4 = Inception(512,(112,24),(160,224,64,64))
        self.incep5 = Inception(512,(128,24),(128,256,64,64))
        self.incep6 = Inception(512,(144,32),(112,288,64,64))
        self.lin_buf21 = nn.Linear(int(528*(3**2)),int(0.2*528*(3**2)))
        self.lin_buf22 = nn.Linear(int(0.2*528*(3**2)),10)
        self.relu4 = nn.ReLU(inplace=True)
        self.incep7 =  Inception(528,(160,32),(256,320,128,128))
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=1)#2
        self.incep8 = Inception(832,(160,32),(256,320,128,128))
        self.incep9 = Inception(832,(192,48),(384,384,128,128))
        #384+384+128+128
        self.dropout = nn.Dropout2d(p=0.4, inplace=False)
        self.dropout1d = nn.Dropout(p=0.6, inplace=False)
        self.pool6 = nn.AvgPool2d(2, stride=1, padding=0)#2
        self.lin1 = nn.Linear(1024,512)
        self.lin2 = nn.Linear(512,10)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.pool1(self.relu1(self.conv1(self.bn1(x))))
        x = self.pool2(self.relu2(self.conv2(self.bn2(x))))
        x = self.incep1.forward(x)
        x = self.incep2.forward(x)
        x = self.pool3(x)
        x = self.incep3.forward(x)
        #h1 = self.lin_buf12(self.dropout1d(self.relu3(self.lin_buf11(x.view(batch_size,int(166*(7**2)))))))
        x = self.incep4.forward(x)
        x = self.incep5.forward(x)
        x = self.incep6.forward(x)
        #h2 = self.lin_buf22(self.dropout1d(self.relu4(self.lin_buf21(x.view(batch_size,int(166*(7**2)))))))
        x = self.incep7.forward(x)
        x = self.pool5(x)
        x = self.incep8.forward(x)
        x = self.incep9.forward(x)
        x = self.dropout(self.pool6(x))
        scores = self.relu5(self.lin1(x.view(batch_size,-1)))

        return scores

def init_weights(m):
    if (type(m) == (nn.Linear))|(type(m)==nn.Conv2d):
        return init.xavier_uniform(m.weight)
    else:
        pass

#if __name__=="__main__":
gpu_dtype = torch.cuda.FloatTensor
torch.cuda.is_available()

NUM_TRAIN = 49000
NUM_VAL = 1000
b_size = 16
TT = T.Compose([T.Resize(256),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=TT)
loader_train = DataLoader(cifar10_train, batch_size=b_size, sampler=ChunkSampler(NUM_TRAIN, 0),num_workers=5)

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=TT)
loader_val = DataLoader(cifar10_val, batch_size=b_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN),num_workers=5)

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                          transform=TT)
loader_test = DataLoader(cifar10_test, batch_size=b_size,num_workers=5)

torch.cuda.synchronize()
model = models.resnet50(pretrained=True) #GoogLeNet()
for t,child in enumerate(model.children()):
    child.requires_grad = False
    for tt,param in enumerate(child.parameters()):
        param.requires_grad = False

model_ext = nn.Sequential(*list(model.children())[:-1])

num_ftrs = model.fc.in_features
model = nn.Sequential(nn.Linear(num_ftrs,100),nn.Linear(100,10))

model.cuda()
model_ext.cpu()

model.is_cuda()


loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train(model, model_ext, loss_fn, optimizer, num_epochs=1)

#check_accuracy(model, loader_val)


'''
    self.bn1 = nn.BatchNorm2d(3,eps=1e-5)
    self.conv1 = nn.Conv2d(3,64,kernel_size=6,stride=2)#14
    self.relu1 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1)#10
    #self.bn2 = nn.BatchNorm2d(64,eps=1e-5)
    self.conv2 = nn.Conv2d(64,192,kernel_size=5,stride=1)#6
    self.relu2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(kernel_size=4,stride=1)# 3
    #self.bn3 = nn.BatchNorm2d(192,eps=1e-5)###########
    self.incep1 = Inception(192,(96,16),(64,128,32,32))
    self.incep2 = Inception(256,(128,32),(128,192,96,64))
    self.pool3 = nn.MaxPool2d(kernel_size=1,stride=1)#3
    self.incep3 = Inception(480,(96,16),(192,208,48,64))
    size_lin = int(512*(3**2))
    self.lin_buf11 = nn.Linear(size_lin,int(0.5*size_lin))
    self.lin_buf12 = nn.Linear(int(0.5*size_lin),10)
    self.relu3 = nn.ReLU(inplace=True)
    self.incep4 = Inception(512,(112,24),(160,224,64,64))
    self.incep5 = Inception(512,(128,24),(128,256,64,64))
    self.incep6 = Inception(512,(144,32),(112,288,64,64))
    self.lin_buf21 = nn.Linear(int(528*(3**2)),int(0.2*528*(3**2)))
    self.lin_buf22 = nn.Linear(int(0.2*528*(3**2)),10)
    self.relu4 = nn.ReLU(inplace=True)
    self.incep7 =  Inception(528,(160,32),(256,320,128,128))
    self.pool5 = nn.MaxPool2d(kernel_size=2,stride=1)#2
    self.incep8 = Inception(832,(160,32),(256,320,128,128))
    self.incep9 = Inception(832,(192,48),(384,384,128,128))
    #384+384+128+128
    self.dropout = nn.Dropout2d(p=0.4, inplace=False)
    self.dropout1d = nn.Dropout(p=0.6, inplace=False)
    self.pool6 = nn.AvgPool2d(2, stride=1, padding=0)#2
    self.lin1 = nn.Linear(1024,512)
    self.lin2 = nn.Linear(512,10)
    self.relu5 = nn.ReLU(inplace=True)
'''
'''
    self.bn1 = nn.BatchNorm2d(3,eps=1e-5)
    self.conv1 = nn.Conv2d(3,32,kernel_size=6,stride=1)#27
    self.relu1 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1)#23
    self.bn2 = nn.BatchNorm2d(32,eps=1e-5)
    self.conv2 = nn.Conv2d(32,64,kernel_size=5,stride=1)#19
    self.relu2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)# 9
    self.bn3 = nn.BatchNorm2d(192,eps=1e-5)###########
    self.incep1 = Inception(64,(25,7),(34,58,12,12))
    self.incep2 = Inception(116,(25,7),(34,58,12,12))
    self.pool3 = nn.MaxPool2d(kernel_size=3,stride=1)#7
    self.incep3 = Inception(116,(60,12),(44,78,22,22))
    size_lin = int(166*(7**2))
    self.lin_buf11 = nn.Linear(size_lin,int(0.5*size_lin))
    self.lin_buf12 = nn.Linear(int(0.5*size_lin),10)
    self.relu3 = nn.ReLU(inplace=True)
    self.incep4 = Inception(166,(60,12),(44,78,22,22))
    self.incep5 = Inception(166,(60,12),(44,78,22,22))
    #self.incep6 =Inception(166,(60,12),(44,78,22,22))
    self.lin_buf21 = nn.Linear(int(166*(7**2)),int(0.2*166*(7**2)))
    self.lin_buf22 = nn.Linear(int(0.2*166*(7**2)),10)
    self.relu4 = nn.ReLU(inplace=True)
    self.incep7 =  Inception(166,(60,12),(44,78,22,22))
    self.pool5 = nn.MaxPool2d(kernel_size=4,stride=1)#4
    self.incep8 =Inception(166,(60,12),(44,78,22,22))
    #self.incep9 = Inception(166,(60,12),(44,78,22,22))
    self.dropout = nn.Dropout2d(p=0.4, inplace=False)
    self.dropout1d = nn.Dropout(p=0.4, inplace=False)
    self.pool6 = nn.AvgPool2d(3, stride=1, padding=0)#2
    self.lin1 = nn.Linear(166*2*2,10)

    self.relu5 = nn.ReLU(inplace=True)
'''
