from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        positive_margins = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                positive_margins += 1
        dW[:, y[i]] -= positive_margins * X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    """
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(num_train), y] = 0

    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    margins[margins > 0] = 1
    positive_counts = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -positive_counts
    dW = X.T.dot(margins) / num_train
    dW += 2 * reg * W

    return loss, dW
