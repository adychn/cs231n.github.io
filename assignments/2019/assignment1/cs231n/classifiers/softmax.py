from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    _, C = W.shape
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)

    for i in range(N):
        softmax = np.exp(scores[i]) / np.sum(np.exp(scores[i]))
        loss += -np.log(softmax[y[i]])
        for j in range(C):
            # dL/dw = dl/df (loss function) *       ->   -y / p
            #         df/dz (activation function)   ->   p_i(1 - p_i) for i==k, -p_i*p_k for i!=k
            #         dL/dz = p_i - y_i
            #         dz/dw (linear weighted sum)   ->   X[i]
            #         (p - y)x = px - yx
            # dL/df * df/dz often generalize as dL/dz, e.g. dL/dz for cross entropy with softmax is p - y.
            # https://www.ics.uci.edu/~pjsadows/notes.pdf
            dW[:, j] += X[i] * softmax[j] # px, softmax[j] is a scaler
        dW[:, y[i]] -= X[i] # -yx where j == y[i]

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + reg * 2 * W  # dL/dW of reg is 2W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    _, C = W.shape

    # scores, minus max for numerical stability
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    # softmax
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    # loss - cross entropy  H(p, q) = -np.sum(p * log(q))
    # p is the expected value = 1 (where y is true), log(q) is the estimated value (our calculation).
    loss = np.sum(-np.log(softmax[np.arange(N), y])) / N
    loss += reg * np.sum(W**2)


    # dL/dsoftmax * dsoftmax/dscores = softmax - y (y = 1 at y[i], = 0 at rest of classes) N x C
    dscores = softmax
    dscores[np.arange(N), y] -= 1
    # dscores/dw     D x C
    dW = X.T.dot(dscores) / N
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
