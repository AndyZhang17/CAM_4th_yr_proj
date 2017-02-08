import autograd.numpy as np 
import autograd.numpy.random as npr
import autograd.numpy.linalg as linalg
import utils_plots as disply
import matplotlib.pyplot as plt
import models as model
from autograd import grad
import utils

PI = model.PI
INF = model.INF

def kl_div( t, var_ard, len_weight, w, s, log_pz, log_qw ):
    # reparameterisation, N x DIM_Z
    z_samples = model.reparam(w, var_ard, len_weight, s, t)
    log_p = log_pz(z_samples)
    print log_p
    log_q = model.log_q_reparam(w, var_ard, len_weight,s,t,log_qw)
    KL_qp = np.mean( log_q - log_p )
    return KL_qp


if __name__ == '__main__':
    # 2-dim example
    DIM_Z = 2  # dimension of z
    MAX_ITER = 5
    LEARNING_RATE = 0.1

    ## defining target dist, p(z)
    PROBS = np.array( [0.7, 0.3] )
    MU_ARR  = np.array([ [1, 1],\
                       [-1, -1]  ])
    VAR_ARR = np.zeros((2,2,2))
    VAR_ARR[0,:,:] = np.eye(2)
    VAR_ARR[1,:,:] = np.eye(2)/2.
    log_pz,_ = utils.gaussian_mix_init(MU_ARR, VAR_ARR, PROBS)

    ## defining the variational model
    PSEUDO_SIZE = 10
    SAMPLING_SIZE = 100

    gp_var = 1
    gp_len = 100
    (gp_s,gp_t) = model.pseudo_data_gen(DIM_Z,PSEUDO_SIZE)

    log_qw, w_gen = utils.uniform_init(-1,+1, dim=DIM_Z)

    w_samples = w_gen(SAMPLING_SIZE,)

    #KL = kl_div(var_ard, len_weight, gp_t)
    #print KL
    print('Forming grad_KL')
    grad_kl_t   = grad(kl_div, argnum=0)
    grad_kl_s   = grad(kl_div, argnum=4)
    grad_kl_var = grad(kl_div, argnum=1)
    grad_kl_len = grad(kl_div, argnum=2)

    print('Calculating grad_KL')
    #gt, gvar, glen = grad_kl(gp_t, var_ard, len_weight,w_samples,gp_s, log_qw)
    kl_distance = kl_div( gp_t, gp_var, gp_len, w_samples, gp_s, log_pz, log_qw )
    print('initial KL: %s' % (kl_distance))


    kl_history = [kl_distance]

    for it in range(MAX_ITER):
        w_samples = w_gen(SAMPLING_SIZE)

        grad_t   = grad_kl_t(  gp_t, gp_var, gp_len,w_samples,gp_s, log_pz, log_qw)
        gp_t   -= LEARNING_RATE * grad_t

        kl_distance = kl_div( gp_t, gp_var, gp_len, w_samples, gp_s, log_pz, log_qw )
        old_kl = kl_history[-1]
        kl_history.append(kl_distance)
        print('iter-%s, DELTA(KL): %.3f, KL: %.3f.' %(it, kl_distance-old_kl, kl_distance))


    z = model.reparam(w_samples, gp_var, gp_len, gp_s, gp_t)
    plt.plot(w_samples, z[:,1], 'b.')
    plt.plot(gp_s, gp_t[:,1],'r-')
    plt.show()




    #kl_history = np.array( kl_history)
    #plt.plot(range(len(kl_history)), kl_history, 'r')

    plt.plot(gp_s, gp_t[:,1],'r--')
    z = model.reparam(w_samples, gp_var, gp_len, gp_s, gp_t)
    plt.plot(w_samples, z[:,1], 'b.')
    plt.show()








