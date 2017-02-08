import autograd.numpy as np
import numpy.random as npr
import autograd.numpy.linalg as linalg

PI = np.pi
INF = 1e+10

def uniform_init(lower, upper,dim=1):
    low, up = min(lower,upper),max(lower,upper)
    dim_x = dim
    width = up-low
    log_unif = -np.log(width)
    def log_p(x):
        out = np.ones(np.shape(x)) *log_unif
        out[x>up], out[x<low] = -INF, -INF
        return out
    def generator(size, dim=dim_x):
        if dim==1:
            return npr.rand(size)*width+low
        return npr.rand(size,dim)*width+low
    return log_p, generator



def gaussian_init(mu_in,var_in):
    # single Gaussian
    mu, var = mu_in, var_in
    d = len(mu)
    if d==1:
        def log_gaussian(x):
            log_p_const = -0.5 *np.log(2*PI) -0.5*np.log(var)
            sub_mu = x-mu
            return log_p_const -0.5*sub_mu*sub_mu/var
        def generator(size):
            return npr.normal(mu,np.sqrt(var),size)
    else:
        var_det, var_inv = linalg.det(var), linalg.inv(var)
        log_p_const = -(d/2.) *np.log(2*PI) -0.5 *np.log(var_det)
        def log_gaussian(x):
            sub_mu = x-mu
            #out = log_p_const - 0.5*np.sum(np.multiply(sub_mu,np.dot(var_inv,sub_mu.T).T ),1)
            out = log_p_const - 0.5*np.sum(np.multiply(sub_mu,np.dot(sub_mu,var_inv.T) ),1)
            return out
        def generator(size):
            return npr.multivariate_normal(mu,var,size)
    return log_gaussian, generator



def gaussian_mix_init(mu_arr, var_arr, prob_arr):
    # default, dimension>1
    gs = list()
    if mu_arr.ndim==1:
        num_g,d = np.shape(mu_arr)[0],1
        gs = [ (gaussian_init(np.array([mu_arr[i]]),np.array(var_arr[i]))) for i in range(num_g) ]
    else:
        num_g,d = np.shape(mu_arr)
        gs = [ (gaussian_init(mu_arr[i,:],var_arr[i,:,:])) for i in range(num_g) ]
    def log_gaussian_mix(x):
        log_gs = np.array([g[0](x) for g in gs]).T
        prob_gs = np.exp(log_gs)
        probs = np.sum( prob_gs *prob_arr, 1)
        return np.log( probs )
    def generator(size):
        indices = np.argmax( npr.multinomial(1,prob_arr,size), axis=1 )
        samples = [ gs[id][1](1)[0] for id in indices  ]
        return np.array(samples)
    return log_gaussian_mix, generator


def spacing_gen( size, low, high, dim=1):
    '''return SIZE x DIM'''
    s = np.linspace( low, high, num=size )
    if dim!=1:
        s = np.reshape( np.tile(s,dim), (dim,size)).T
    return s