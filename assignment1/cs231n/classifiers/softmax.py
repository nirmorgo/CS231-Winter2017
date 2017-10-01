import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  num_classes = W.shape[1]
    
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_score = np.exp(scores[y[i]])
    scores_sum = 0
    dWi = np.zeros_like(W)
    
    for j in range(num_classes):
      scores_sum += np.exp(scores[j])
      dWi[:,j] += np.exp(scores[j]) * X[i]
    
    loss += -np.log(correct_score / scores_sum)
    dWi /= scores_sum
    dWi[:,y[i]] -= X[i]
        
    dW += dWi
  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W
  
  
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
  num_classes = W.shape[1]

  scores = np.exp(X.dot(W))
  correct_scores = scores[range(num_train),y]
  scores_sums = np.sum(scores,axis=1)  
  loss -= np.sum(np.log(correct_scores/scores_sums))
    
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  # duplicate scores sum so we can normalize it with scores
  scores_sums = np.tile(np.array([scores_sums]).T,(1,num_classes))
  scores = scores/scores_sums
    
  # add weight of the correct class to the scores
  scores[range(num_train),y] -= 1

  dW += (X.T).dot(scores)
    
  dW /= num_train
  dW += reg * 2 * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

