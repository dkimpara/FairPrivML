import numpy


'''Private SGD from song et al'''

def SGDPriv(f, x0, fprime, X, y, s, eps, lam):
    sumloss = 0
    coef = x0
    coef_size = size(x0)
    optimal_init = 1.0 / lam
    
    for i in range(size(y)): #batch size = 1
        eta = 1.0 / (lam * (optimal_init + t - 1)) #optimal learning rate

        loss = (coef, X, y, s)
        grad = fprime(coef, X, y, s, i)

        noise = numpy.random.laplace(loc = 0.0, scale = 2 / epsilon,
                                     coef_size)
        #clip gradient with l_2 norm
        grad = grad / max(1, numpy.linalg.norm(grad))
        
        #update weights
        coef -= eta * (lam * coef + grad + noise)

        print(loss)
    return coef    

