import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes =  W.shape[1]
  for i in range(num_train):
    # loss
    scores = X[i].dot(W) # 1 x C
    scores -= np.max(scores) # for numeric stability, see the leture notes
    softmax = np.exp(scores)/np.sum(np.exp(scores)) # 1 x C
    loss += -np.log(softmax[y[i]])
    # gradient
    softmax[y[i]] -= 1
    dW += np.outer(X[i], softmax)
    # Above equals np.dot(X[i].T, softmax), but np.dot(X[i].T, softmax) doesn't
    # work in numpy, use below code as replacement.
    # dW += np.dot(np.reshape(X[i], (-1,1)), np.reshape(softmax, (1, -1)))
  # loss
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  # gradient
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  arange = np.arange(num_train)
  # loss
  scores = X.dot(W) # N x C
  scores -= np.reshape(np.max(scores, axis=1), (-1, 1)) # for numeric stability
  softmax = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (-1, 1))
  loss = np.sum(-np.log(softmax[arange, y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  # gradient
  softmax[arange, y] -= 1 # N x C
  dW = np.dot(X.T, softmax) # Try to understand this vectorized formula
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

