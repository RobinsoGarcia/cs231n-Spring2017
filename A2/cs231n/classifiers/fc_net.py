from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        W1 = np.random.randn(input_dim,hidden_dim)*weight_scale
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim,num_classes)*weight_scale
        b2 = np.zeros(num_classes)
        self.params={'W1':W1,'b1':b1,'W2':W2,'b2':b2}


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        h1,cacheh1 = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        scores,cacheh2 = affine_forward(h1,self.params['W2'],self.params['b2'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dl = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])



        dx, grads['W2'], grads['b2'] = affine_backward(dl, cacheh2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dx, cacheh1)

        grads['W2'] = grads['W2']+self.reg*self.params['W2']
        grads['W1'] = grads['W1']+self.reg*self.params['W1']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, activation='relu',batch_size=100,Xavier=False, loss_function='softmax',dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        - Xavier: if true, utilizes Xavier initialization.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.activation = activation
        self.loss_function = loss_function
        self.batch_size=batch_size

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        if Xavier==True:
            def build_block(layer,dims):
                self.params['W'+str(layer)] = np.random.randn(size,dims)*(1/np.sqrt((size+dims)))
                self.params['b'+str(layer)] = np.zeros(dims)

                if self.activation=='leaky_relu':
                    self.params['alpha'+str(layer)] = np.float32(0.01)
                elif self.activation=='PRelu':
                    self.params['alpha'+str(layer)] = np.ones((batch_size,dims))*(1/np.sqrt((size+dims)))
                #else:
                #    self.params['alpha'+str(layer)]= np.float32(0)

                if self.use_batchnorm ==True:
                    self.params['gamma'+str(layer)] = np.ones(dims)
                    self.params['beta'+str(layer)] = np.zeros(dims)
        else:
            def build_block(layer,dims):
                self.params['W'+str(layer)] = np.random.randn(size,dims)*weight_scale
                self.params['b'+str(layer)] = np.zeros(dims)
                
                if self.activation=='leaky_relu':
                    self.params['alpha'+str(layer)] = np.float32(0.01)
                elif self.activation=='PRelu':
                    self.params['alpha'+str(layer)] = np.ones((batch_size,dims))*0.01
                #else:
                #    self.params['alpha'+str(layer)] = np.float32(0)

                
                if self.use_batchnorm ==True:
                    self.params['gamma'+str(layer)] = np.ones(dims)
                    self.params['beta'+str(layer)] = np.zeros(dims)
            

            
        size = input_dim
        for i in np.arange(self.num_layers-1):
            layer = i+1
            build_block(layer,hidden_dims[i])
            size = hidden_dims[i]

        if Xavier==True:
            self.params['W'+str(self.num_layers)] = np.random.randn(size,num_classes)*(1/np.sqrt((size+dims)))
        else:
            self.params['W'+str(self.num_layers)] = np.random.randn(size,num_classes)*weight_scale

        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
    


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
    
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        self.cache = {}
        h = X
        
        for i in np.arange(self.num_layers-1):
            w = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)] 
            if self.use_batchnorm==True:
                gamma = self.params['gamma'+str(i+1)]
                beta = self.params['beta'+str(i+1)]
                
                if self.activation =='relu':
                    h, cache =  affine_batchnorm_relu_forward(h, w, b, gamma, beta,self.bn_params[i])
                
                elif (self.activation =='leaky_relu') | (self.activation =='PRelu'):
                    alpha = self.params['alpha'+str(i+1)]
                    h, cache =  affine_batchnorm_leaky_relu_forward(h, w, b, gamma, beta,self.bn_params[i],alpha)
                
                self.bn_params[i] = cache[1][-1]
                self.cache['affine_batchnorm_relu'+str(i+1)] = cache
                if self.use_dropout:
                    h,cache = dropout_forward(h, self.dropout_param)
                    self.cache['dropout'+str(i+1)] = cache

            else:
                if self.activation == 'relu':
                    h,self.cache['affine_relu'+str(i+1)] = affine_relu_forward(h, w, b)
                elif (self.activation =='leaky_relu') | (self.activation =='PRelu'):
                    alpha = self.params['alpha'+str(i+1)]
                    h,self.cache['affine_relu'+str(i+1)] = affine_leaky_relu_forward(h, w, b,alpha)
                
                if self.use_dropout:
                    h,cache = dropout_forward(h, self.dropout_param)
                    self.cache['dropout'+str(i+1)] = cache

        w = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores, cache = affine_forward(h, w, b)
        self.cache['cache_affine'+str(i+2)] = cache


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, self.grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        if self.loss_function=='softmax':
            loss, dl = softmax_loss(scores, y)
        elif self.loss_function=='svm':
            loss, dl = svm_loss(scores, y)
            
        
        def update_grads(dx,dw,db,layer,dgamma,dbeta,dalpha=0.01):
            dw += self.reg*self.params['W'+str(layer)]
            self.grads['W'+str(layer)] = dw
            self.grads['b'+str(layer)] = db
            self.grads['gamma'+str(layer)] = dgamma 
            self.grads['beta'+str(layer)] = dbeta
            if self.activation=='PRelu':
                self.grads['alpha'+str(layer)] = dalpha
            
            elif self.activation=='leaky_relu':
                self.grads['alpha'+str(layer)] = 0

            #elif (self.activation=='relu')|(self.activation=='leaky_relu'):
            #    self.grads['alpha'+str(layer)] = 0
            pass

        def update_affine_grads(x,dw,db,layer,dalpha=0.01):
            dw += self.reg*self.params['W'+str(layer)]
            self.grads['W'+str(layer)] = dw
            self.grads['b'+str(layer)] = db
            if self.activation=='PRelu':
                self.grads['alpha'+str(layer)] = dalpha
            if self.activation=='leaky_relu':
                self.grads['alpha'+str(layer)] = 0

            #if (self.activation=='relu')|(self.activation=='leaky_relu'):
            #    self.grads['alpha'+str(layer)] = 0
            pass
        
 
        reg1 = 0
        for i in np.arange(self.num_layers)+1:
            reg1 += np.sum(self.params['W'+str(i)]**2)
            
            
        loss += 0.5*self.reg*reg1
        dout = dl 

        dout, dw, db = affine_backward(dout,self.cache['cache_affine'+str(self.num_layers)])
        update_affine_grads(dout,dw,db,layer=self.num_layers)
        for i in np.flip(np.arange(self.num_layers-1)+1,axis=0):
            if self.use_batchnorm==True:
                if self.use_dropout:
                    cache = self.cache['dropout'+str(i)]
                    dout = dropout_backward(dout, cache)
                cache = self.cache['affine_batchnorm_relu'+str(i)]
                if self.activation=='relu':
                    dout, dw, db,dgamma,dbeta = affine_batch_norm_relu_backward(dout, cache)
                    update_grads(dout,dw,db,i,dgamma,dbeta)
                elif (self.activation=='leaky_relu')|(self.activation=='PRelu'):
                    dout, dw, db,dgamma,dbeta,dalpha = affine_batch_norm_leaky_relu_backward(dout, cache)
                    update_grads(dout,dw,db,i,dgamma,dbeta,dalpha)

                
            else:
                if self.use_dropout:
                    cache = self.cache['dropout'+str(i)]
                    dout = dropout_backward(dout, cache)
                cache = self.cache['affine_relu'+str(i)]
                if self.activation=='relu':
                    dout,dw,db = affine_relu_backward(dout, cache)
                    update_affine_grads(dout,dw,db,i)
                elif (self.activation=='leaky_relu')|(self.activation=='PRelu'):
                    dout,dw,db,dalpha = affine_leaky_relu_backward(dout, cache)
                    update_affine_grads(dout,dw,db,i,dalpha)
       

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, self.grads


def affine_batchnorm_relu_forward(x, w, b,gamma,beta,bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    h, cache_affine = affine_forward(x, w, b)
    h, cache_batchnorm = batchnorm_forward(h, gamma, beta, bn_param)
    out, cache_relu = relu_forward(h)
    cache = (cache_affine,cache_batchnorm,cache_relu)
    return out, cache


def affine_batch_norm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    cache_affine,cache_batchnorm,cache_relu = cache
    dout = relu_backward(dout, cache_relu)
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache_batchnorm)
    dx, dw, db = affine_backward(dout, cache_affine)
    return dx, dw, db,dgamma,dbeta

def affine_batchnorm_leaky_relu_forward(x, w, b,gamma,beta,bn_param,alpha):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    h, cache_affine = affine_forward(x, w, b)
    h, cache_batchnorm = batchnorm_forward(h, gamma, beta, bn_param)
    out, cache_relu = leaky_relu_forward(h,alpha)
    cache = (cache_affine,cache_batchnorm,cache_relu)
    return out, cache


def affine_batch_norm_leaky_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    cache_affine,cache_batchnorm,cache_relu = cache
    dout,dalpha = leaky_relu_backward(dout, cache_relu)
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache_batchnorm)
    dx, dw, db = affine_backward(dout, cache_affine)
    return dx, dw, db,dgamma,dbeta,dalpha

def affine_leaky_relu_forward(x, w, b,alpha):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    h, cache_affine = affine_forward(x, w, b)
    out, cache_relu = leaky_relu_forward(h,alpha)
    cache = (cache_affine,cache_relu)
    return out, cache


def affine_leaky_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    cache_affine,cache_relu = cache
    dout,dalpha = leaky_relu_backward(dout, cache_relu)
    dx, dw, db = affine_backward(dout, cache_affine)
    return dx, dw, db,dalpha

def leaky_relu_forward(x,alpha):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(alpha*x,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x,alpha
    
    return out, cache


def leaky_relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x,alpha = cache
    
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    mask1 = (alpha*x<x)
    
    mask2 =  (alpha*x>=x)

    dx = (mask1*1+mask2*alpha)*dout
  

    dalpha = dout*mask2*x
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx,dalpha