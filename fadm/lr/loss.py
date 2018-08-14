import numpy

def sigmoid(x, w):
    """ sigmoid(w^T x)
    To suppress the warnings at np.exp, do "np.seterr(all='ignore')

    Parameters
    ----------
    x : array, shape=(d)
        input vector
    w : array, shape=(d)
        weight

    -------
    sigmoid : float
        sigmoid(w^T x)
    """

    s = np.clip(np.dot(w, x), -SIGMOID_RANGE, SIGMOID_RANGE)

    return 1.0 / (1.0 + np.exp(-s))

def loss(coef, x, y, s):

def grad_loss(coef, x, y, s, n_sample, n_0, n_1):

    
    grad_fair = n_sample * (s / n_1 - (1 - s) / n_0)
                * sigmoid(x, coeff) * (1 - sigmoid(x, coeff))
    grad_loss = y * sigmoid(x, coeff) * (1 - sigmoid(x, coeff)) +
                (1 - y) * sigmoid(x, coeff) * (1 - sigmoid(x, coeff))

    return -grad_loss + self.eta * grad_fair + self.C * coeff


    
