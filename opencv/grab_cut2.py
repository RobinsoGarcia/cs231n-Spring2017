import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import random
%matplotlib inline
from scipy.ndimage.filters import gaussian_filter1d
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from PIL import Image
gpu_dtype = torch.cuda.FloatTensor
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def preprocess(img, size=224):
    transform = T.Compose([
        T.Scale(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

# Download and load the pretrained SqueezeNet model.
#model = torchvision.models.squeezenet1_1(pretrained=True)
model = torchvision.models.vgg16(pretrained=True)
model.cuda()
# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

from cs231n.data_utils import load_imagenet_val
X, y, class_names = load_imagenet_val(num=5)

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Wrap the input tensors in Variables
    X_var = Variable(X.type(gpu_dtype), requires_grad=True)
    y_var = Variable(y.type(torch.cuda.LongTensor))
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with a backward pass.               #
    ##############################################################################

    scores = model(X_var).gather(1,y_var.view(-1,1)).squeeze()

    scores.backward(torch.ones(scores.size()).type(gpu_dtype))

    saliency = X_var.grad.abs()

    saliency,_ = saliency.max(dim=1)

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency.data.squeeze().cpu()

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)
model.cuda()
saliency = compute_saliency_maps(X_tensor, y_tensor, model)

def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)
    model.cuda()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()

    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

def grabCut_fromSaliency(X,saliency):
    for idx in range(len(X)):
        results = {}
        raw_mask = saliency[idx].numpy()
        img = X[idx]
        results['original']=img
        results['raw_mask']=raw_mask

        max = np.max(raw_mask.flatten())
        mean = np.mean(raw_mask.flatten())
        std = np.std(raw_mask.flatten())

        min = np.min(raw_mask.flatten())

        mask_min = raw_mask < min#+0.0001
        Th_max = mean+2*std



        _,mask_max = cv.threshold(raw_mask,Th_max,1,cv.THRESH_BINARY)

        results['certain_Background'] = mask_min
        results['certain_Foreground'] = mask_max

        mask = np.ones(raw_mask.shape) - mask_min - mask_max

        mask = np.uint8(mask_min*0 + mask_max*3 + mask*2)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask, bgdModel, fgdModel=cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]

        results['grabCut_result'] = img

        plt.figure(figsize=(12, 6))
        for t,i in enumerate(results):
            plt.subplot(1, 6, t + 1)
            plt.imshow(results[i])
            plt.title(i)
            plt.axis('off')
        plt.gcf().tight_layout()

grabCut_fromSaliency(X,saliency)
