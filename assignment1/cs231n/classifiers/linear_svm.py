import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    Li = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        Li += margin
        dW[:,j] += X[i]    # Grad for the wrong classes
        dW[:,y[i]] -= X[i]    # Grad of the correct class
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg *2* W  

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1] 
  num_train = X.shape[0]
  scores = X.dot(W)

  correct_class_scores = scores[range(num_train),y]
  # duplicate correct score so we can deduct it from other scores
  correct_class_scores = np.tile(np.array([correct_class_scores]).T,(1,num_classes))

  margins = np.maximum(0,scores - correct_class_scores +1)
  # make sure that correct scores will not be part of the sum
  margins[range(num_train),y] = 0
    
  loss = np.sum(margins)/num_train + reg*np.sum(W*W)
    
  
    
          
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
    

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
 
  # a binary matrix for the gradient calculation
  thresholds = (margins>0).astype(int) 
  correct_scores_weights = np.sum(thresholds,axis=1)
  thresholds[range(num_train),y] = -correct_scores_weights
    
  dW += (thresholds.T).dot(X).T
  dW /= num_train
  dW += 2 * reg * W
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
