import numpy as np

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

def loss(coef, x, y, s): #todo: access to n_s
    pred = sigmoid(x, coef) #probability of predicting 1 given x

    logLoss = y * pred + (1.0 - y) * (1 - pred)
    fairLoss = n_sample * (s / n_1 - (1 - s) / n_0) * pred
    regLoss = np.linalg.norm(coef)

    return -logLoss + self.eta * fairLoss + self.C * 0.5 * regLoss
def grad_loss(coef, x, y, s, n_sample, n_0, n_1):
    pred = sigmoid(x, coeff)

    grad_fair = x * n_sample * (s / n_1 - (1 - s) / n_0)
                * pred * (1 - pred))

    grad_loss = x * y * pred * (1 - pred) +
                x * (1 - y) * (-pred) * (1 - pred)

    return -grad_loss + self.eta * grad_fair + self.C * coef
