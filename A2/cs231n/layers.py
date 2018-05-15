from builtins import range
import numpy as np

x = np.random.random((10,10,10))
d = x.shape
y = x.reshape(np.prod(d))
y.reshape(d).shape


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """

    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    x_ = x.reshape(x.shape[0],np.prod(x.shape[1:]))
    out = x_.dot(w)+b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    x_shape = x.shape

    x = x.reshape(x_shape[0],np.prod(x_shape[1:]))

    dw = x.T.dot(dout)

    dx = dout.dot(w.T).reshape(x_shape)

    db = np.sum(dout*np.ones(b.shape)[np.newaxis,:],axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
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
    out = np.maximum(x,0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mask = np.maximum(x,0)

    mask[mask>0]=1

    dx = np.multiply(mask,dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx




def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)num_pos
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        sample_mean = np.mean(x,axis=0)
        sample_var = np.std(x,axis=0)**2
  

        x_ = (x - sample_mean[np.newaxis,:])/np.sqrt(sample_var[np.newaxis,:]+eps)
        x_ = gamma*x_ + beta

        out = x_
        cache = (x,gamma,beta,sample_mean,sample_var,eps,bn_param)
      

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var



        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        sample_mean = np.mean(x,axis=0)
        sample_var = np.std(x,axis=0)**2

 
        x_ = (x - running_mean[np.newaxis,:])/np.sqrt(running_var[np.newaxis,:]+eps)
        x_ = gamma*x_ + beta

        out = x_
        
        cache = (x,gamma,beta,sample_mean,sample_var,eps,bn_param)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    '''x: Data of shape (N, D)'''
    x,gamma,_,sample_mean,sample_var,eps,_ = cache


    if len(x.shape)==2:
        N = x.shape[0]
        ax = 0
        dout_shape = dout.shape
    elif len(x.shape)==4:
        N = x.shape[2]
        ax = 2
        dout_shape = dout.shape[2:]
   
    x_ = (x-sample_mean)/np.sqrt(sample_var+eps)

    dbeta = np.sum(np.ones(dout_shape)*dout,axis=ax)
    
    dgamma = np.sum(x_*dout,axis=ax)

    dx1 = gamma*dout/np.sqrt(sample_var+eps)

    dx2 = np.tile(np.sum(-1*gamma*dout/np.sqrt(sample_var+eps),axis=ax),(N,1))/N

    dldvar= np.sum(-1*(x-sample_mean)*np.sqrt(sample_var+eps)**(-2)*gamma*dout,axis=ax)
    
    grad1 = 1/(2*np.sqrt(sample_var))
    
    grad2 = 2*(x-sample_mean)/N
  
    dx3 = np.tile(dldvar*grad1,(N,1))*grad2

    dx4 = np.tile(np.sum(-dx3,axis=ax),(N,1))
 
    dx = dx1+dx2+dx3+dx4

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    x,gamma,_,sample_mean,sample_var,eps,_ = cache

    dxhat = dout*gamma
    x_hat = (x-sample_mean)/np.sqrt(sample_var)
    inv_var = 1./np.sqrt(sample_var+eps)
    
    N = x.shape[0]

    dx = (1./N)*inv_var*(N*dxhat-np.sum(dxhat,axis=0)
    -x_hat*np.sum(dxhat*x_hat,axis=0))
    dgamma = np.sum(x_hat*dout,axis=0)

    dbeta = np.sum(dout,axis=0)
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape)<p
        out = x*mask/p
        cache = (p,mask)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        
        dx = dout*mask/dropout_param['p']
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def im2col(x,conv_param):
        
        N,C,H,W,_,Hh,Ww,s,pad,_,_ = conv_param #['indexes']

        w_conv = np.arange(0,W,s)
        h_conv = np.arange(0,H,s)

        npad = ((0,0),(0,0),(pad,pad),(pad,pad))
        x_pad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
       
        X_col = []
        for i in h_conv:
            for j in w_conv:
            
                v = np.reshape(x_pad[:,:,i:i+Hh,j:j+Ww],(N,C*Hh*Ww))
                X_col.append(v) 
         
        X_col = np.array(X_col)
    

        return X_col

    
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ########################################################################### 

    N,C,H,W = x.shape    
    F,_,Hh,Ww = w.shape
    s = conv_param['stride']
    pad = conv_param['pad']
    
    H2,W2 = (H-Hh+2*pad)/s+1,(W-Ww+2*pad)/s+1
    
    H2 = int(H2)
    W2 = int(W2)

    conv_param['indexes']=[N,C,H,W,F,Hh,Ww,s,pad,H2,W2]

    X_col = im2col(x,conv_param['indexes'])

    W_col = np.reshape(w,(F,C*Hh*Ww))
    
    W_col = np.rollaxis(np.array(W_col),0,2)[np.newaxis,:,:]
    out = np.matmul(X_col,W_col)+b
    out = np.rollaxis(out,0,3)

    out = np.reshape(out,(N,F,H2,W2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, _, conv_param = cache
    N,C,H,W,F,Hh,Ww,s,pad,H2,W2 = conv_param['indexes']

    '''db'''
    db = np.sum(dout,axis=(0,2,3))

    '''dw'''
    X_col = im2col(x,conv_param['indexes'])
    X_col = np.swapaxes(X_col,0,1)
    V = np.reshape(dout,(N,F,H2*W2))
    dw = np.matmul(V,X_col)
    dw = np.sum(dw,axis=0)
    dw = np.reshape(dw,(w.shape))

    '''dx'''
    W_col = np.reshape(w,(F,C,Hh*Ww))  
    W_col = np.flip(W_col,axis=2)
    W_col = np.rollaxis(np.array(W_col),0,2)[np.newaxis,:,:]
    
    '''transpose convolution arithimetics'''
    if s>1:
        num_zeros_2add = s -1
        i_ =  int(H2+(H2-1)*num_zeros_2add)
        idx = np.arange(0,i_,num_zeros_2add+1)
        
        dout_ = np.zeros((N,F,i_,i_))
        
        Ns = np.arange(N)
        Fs = np.arange(F)
        dout_[np.ix_(Ns,Fs,idx,idx)] = dout
        
    else:
        i_ = H2
        dout_=dout
    pad_ = Hh-pad-1 
           
    npad = ((0,0),(0,0),(pad_,pad_),(pad_,pad_))
    v_pad = np.pad(dout_, pad_width=npad, mode='constant', constant_values=0)

     
    r = v_pad.shape[2] - Hh + 1
    c = v_pad.shape[3] - Hh + 1
    s_ = 1
    
    V= []
    for i in range(r):
        for j in range(c):
            v = np.reshape(v_pad[:,:,i*s_:i*s_+Hh,j*s_:j*s_+Ww],(v_pad.shape[0],v_pad.shape[1]*Hh*Ww))      
            V.append(v) 


    V = np.array(V)

    V = np.swapaxes(V,0,1)
  
    W_col = np.reshape(W_col,(1,C,F*Hh*Ww))
    dx = np.matmul(W_col,np.swapaxes(V,1,2))
    dx = np.reshape(dx,(N,C,H,W))
 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N,C,H,W = x.shape
    Hh = pool_param['pool_height']
    Ww = pool_param['pool_width']
    s = pool_param['stride']
    o = int((H-Hh)/s+1)
    
 
    w_conv = np.arange(0,W,s)
    h_conv = np.arange(0,H,s)

    V = []
    for i in h_conv:
        for j in w_conv:
            v = np.reshape(x[:,:,i:i+Hh,j:j+Ww],(N,C,Hh*Ww))
            v = np.max(v,axis=2)           
            V.append(v) 
    
    V = np.rollaxis(np.array(V),0,3)
    V = np.reshape(V,(N,C,o,o))
    
    out = V
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x,pool_param = cache
    
    N,C,H,W = x.shape
    Hh = pool_param['pool_height']
    Ww = pool_param['pool_width']
    s = pool_param['stride']

    dx = np.zeros_like(x)
 
    w_conv = np.arange(0,W,s)
    h_conv = np.arange(0,H,s)

    row=0
    col=0
    for n in range(N):
        for k in range(C):
            for i in h_conv:
                for j in w_conv:
                    x_ = x[n,k,i:i+Hh,j:j+Ww]
                    mask = (x_ == np.max(x_))
                    dx[n,k,i:i+Hh,j:j+Ww] = mask*dout[n,k,row,col]
                    col+=1
                row+=1
                col=0
            row=0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    
    
    eps = bn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    gamma = gamma[np.newaxis,:,np.newaxis,np.newaxis]
    beta = beta[np.newaxis,:,np.newaxis,np.newaxis]

    sample_mean = np.mean(x,axis=(2))[:,:,np.newaxis,:]
    sample_var = np.std(x,axis=(2))[:,:,np.newaxis,:]**2
  
    out = gamma*(x-sample_mean)/np.sqrt(sample_var+eps) + beta

    cache = (x,gamma,beta,sample_mean,sample_var,eps,bn_param)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    x,gamma,_,sample_mean,sample_var,eps,b_ = cache

    
    dxhat = dout*gamma
    x_hat = (x-sample_mean)/np.sqrt(sample_var+eps)
    inv_var = 1./np.sqrt(sample_var+eps)
    N = x.shape[2]

    dx = (1./N)*inv_var*(N*dxhat-np.sum(dxhat,axis=2)[:,:,np.newaxis,:]
    -x_hat*np.sum(dxhat*x_hat,axis=2)[:,:,np.newaxis,:])

    dgamma = np.sum(x_hat*dout,axis=(0,2,3))

    dbeta = np.sum(dout,axis=(0,2,3))
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
