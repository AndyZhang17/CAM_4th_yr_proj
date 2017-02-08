import autograd.numpy.random as npr
import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
from autograd import grad
import utils as utils
import utils_plots as disply

PI = np.pi
INF = 1e+10
DELTA = 1e-8

## everything in 1-dimension now

def reparam(w, A, B, C, L, P, indep=False):
    '''
    mapping w->z, K->K
    :param w: N x K
    :B,L,C: #_sigmoid
    :return:  N x K
    '''
    if w.ndim==1:
        w_ = w.reshape((1,len(w)))
        A_ = np.exp(A)
        L_ = np.exp(L)
        components = A_*( 1./( 1+np.exp(-L_*(w_.T-B)) ) + C )
        if indep:
            return components
        return np.sum( components*P, 1 )
    else:
        pass

## TODO testing
df_dw = ele_grad(reparam, argnum=0)
# return: N


def params_init(num_sig, mode='linear'):
    A = np.array([4.]*num_sig)
    B = np.array([0.]*num_sig)
    C = np.array([-0.5]*num_sig)
    L = np.array([1.]*num_sig)
    P = np.array([1./num_sig]*num_sig)
    if mode.upper()=='LINEAR':
        pass
    elif mode.upper()=='RANDOM':
        pass
        A = A + (npr.rand(num_sig)-0.5)*3
        B = B + (npr.rand(num_sig)-0.5)*10
        C = C + (npr.rand(num_sig)-0.5)*2
        L = L + (npr.rand(num_sig)-0.5)*4
        P = npr.dirichlet([1.]*num_sig,1)[0]
        A[A<DELTA], L[L<DELTA] = DELTA, DELTA
    else:
        pass
    A = np.log(A)
    L = np.log(L)
    return A,B,C,L,P



def log_qz( w, A,B,C,L,P,log_qw ):
    if w.ndim==1:
        dfdw = df_dw(w,A,B,C,L,P)
        out = log_qw(w) - np.log(np.abs(dfdw))
        return out
    else:
        pass

def plot_qz( A,B,C,L,P,log_qw, target=None, testing=False):
    ## 1-dimension case
    w_samples = utils.spacing_gen(1000,-1,+1,dim=1)

    z_probs = np.exp( log_q(w_samples, A,B,C,L,P,log_qw) )
    z_samples = reparam(w_samples,A,B,C,L,P)
    if testing:
        idx = np.argsort(z_samples)
        z_samples = z_samples[idx]
        z_probs = z_probs[idx]
        NORM = 0
        for i in range(len(z_samples)-1):
            NORM += (z_samples[i+1]-z_samples[i])*(z_probs[i+1]+z_probs[i])/2.
        print('checking q-dist norm: %.6f' % (NORM))

    if target==None:
        disply.line_2d(z_samples,z_probs,linetype='.',ylims=(0,1))
    else:
        tar_probs = np.exp(target(z_samples))
        probs = np.array([tar_probs,z_probs]).T
        disply.line_2d(z_samples,probs,linetype='.',ylims=(0,1))


# def kl_div( w_samples, log_target, log_qw, A,B,C,L,P ):
#     z = reparam(w_samples, A,B,C,L,P)
#     log_p = log_target(z)
#     log_q = log_qz(w_samples, A,B,C,L,P,log_qw)
#     KL = np.mean( log_q-log_p )
#     return KL
#dkl_dA = grad(kl_div,argnum=3)

def kl_div( w_samples, log_target, log_qw, params ):
    A,B,C,L,P = params['A'],params['B'],params['C'],params['L'],params['P'],
    z = reparam(w_samples, A,B,C,L,P)
    log_p = log_target(z)
    log_q = log_qz(w_samples, A,B,C,L,P,log_qw)
    KL = np.mean( log_q-log_p )
    return KL

dkl_dA = grad(kl_div,argnum=3)


def grad_kl(w_samples,log_target,log_qw, A,B,C,L,P):
    params = {'A':A,'B':B,'C':C,'L':L,'P':P}
    #return dkl_dA(w_samples,log_target,log_qw,A,B,C,L,P)
    return dkl_dA(w_samples,log_target,log_qw,params)
