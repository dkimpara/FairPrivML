import numpy
from loss import loss, grad_loss


'''Private SGD from song et al'''

def SGDPriv(x0, X, y, s, eps, lam):
    sumloss = 0
    coef = x0
    coef_size = size(x0)
    optimal_init = 1.0 / lam
    
    for i in range(size(y)): #batch size = 1
        nu = 1.0 / (lam * (optimal_init + t - 1)) #optimal learning rate

        sumloss += loss(coef, X[i], y[i], s[i])
        grad = grad_loss(coef, X[i], y[i], s[i])

        noise = numpy.random.laplace(loc = 0.0, scale = 2 / epsilon,
                                     coef_size)
        #clip gradient with l_2 norm
        grad = grad / max(1, numpy.linalg.norm(grad))
        
        #update weights
        coef -= nu * (lam * coef + grad + noise)

        print(loss)
    return coef    

