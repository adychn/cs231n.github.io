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
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        # Z1 = X.dot(W1) + b1
        # Out1 = np.maximum(0, Z1)
        # Z2 = Out1.dot(W2) + b2

        Z1, Z1_cache = affine_relu_forward(X, W1, b1)
        Z2, Z2_cache = affine_forward(Z1, W2, b2)

        scores = Z2 # log odds

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Returns a tuple of:
        # - loss: Scalar giving the loss
        # - dx: Gradient of the loss with respect to x
        loss, dZ2 = softmax_loss(Z2, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        # dL/dW2 = dL/dZ * dZ/dW2
        # Returns a tuple of:
        # - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        # - dw: Gradient with respect to w, of shape (D, M)
        # - db: Gradient with respect to b, of shape (M,)
        dZ1, dW2, db2 = affine_backward(dZ2, Z2_cache)
        _, dW1, db1 = affine_relu_backward(dZ1, Z1_cache)

        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # input     W1(layer1 relu)      W2(layer2 relu)      FC(layer3)      loss
        layer_dims = [input_dim]
        layer_dims.extend(hidden_dims)
        layer_dims.append(num_classes)
        # print(layer_dims)
        for i in range(self.num_layers):
            W = "W" + str(i + 1)
            b = "b" + str(i + 1)
            self.params[W] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i+1])
            self.params[b] = np.zeros(layer_dims[i+1])
        # print(self.params.keys())

        # Batch norm and Layer norm params
        if self.normalization != None:
            for i in range(self.num_layers - 1):
                self.params['gamma'+str(i+1)] = np.ones(layer_dims[i+1])
                self.params['beta'+str(i+1)] = np.ones(layer_dims[i+1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

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
        if self.normalization=='batchnorm':
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = []
        x = X

        # first to one before last layers
        for i in range(1, self.num_layers):
            w = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                bn_param = self.bn_params[i-1]
                x, cache = affine_norm_relu_forward(x, w, b, gamma, beta, bn_param, \
                    self.normalization, self.use_dropout, self.dropout_param)
            else:
                x, cache = affine_relu_forward(x, w, b)
            caches.append(cache)

        # last layer
        w = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        scores, cache = affine_forward(x, w, b)
        caches.append(cache)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # loss
        loss, dscores = softmax_loss(scores, y)
        W = 0
        for i in range(self.num_layers):
            W += np.sum(self.params['W'+str(i+1)] ** 2)
        loss += 0.5 * self.reg * W

        # gradients
        # - cache: (x, w, b)
        cache_idx = len(caches) - 1

        W = 'W' + str(self.num_layers)
        b = 'b' + str(self.num_layers)
        dZ, dW, db = affine_backward(dscores, caches[cache_idx])
        grads[W] = dW + self.reg * self.params[W]
        grads[b] = db

        cache_idx -= 1
        for i in range(self.num_layers - 1, 0, -1):
            assert cache_idx >= 0
            if self.normalization=='batchnorm' or self.normalization == 'layernorm':
                dZ, dW, db, dgamma, dbeta = affine_norm_relu_backward(\
                            dZ, caches[cache_idx], self.normalization, self.use_dropout)
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
            else:
                dZ, dW, db = affine_relu_backward(dZ, caches[cache_idx])
            grads['W'+str(i)] = dW + self.reg * self.params['W'+str(i)]
            grads['b'+str(i)] = db
            cache_idx -= 1


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

# Additional helper method created by me.
def affine_norm_relu_backward(dout, cache, normalization, dropout):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache, dropout_cache = cache

    if dropout:
        dout = dropout_backward(dout, dropout_cache)

    da = relu_backward(dout, relu_cache)

    if normalization == 'batchnorm':
        dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    elif normalization == 'layernorm':
        dbn, dgamma, dbeta = layernorm_backward(da, bn_cache)

    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param, normalization, use_dropout, dropout_param):
    """
    affine - batch/layer norm - relu - dropout

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc_cache, bn_cache, relu_cache, dropout_cache = None, None, None, None

    a, fc_cache = affine_forward(x, w, b)

    if normalization == 'batchnorm':
        z, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    elif normalization == 'layernorm':
        z, bn_cache = layernorm_forward(a, gamma, beta, bn_param)

    dout, relu_cache = relu_forward(z)

    if use_dropout:
        dout, dropout_cache = dropout_forward(dout, dropout_param)


    cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
    return dout, cache
