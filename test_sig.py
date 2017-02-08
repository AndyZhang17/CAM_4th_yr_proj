import matplotlib.pyplot as plt
import utils_plots as disply
import autograd.numpy as np
import utils as utils
import func_sigmoid as sig
from autograd.optimizers import adam



# Testing the mapping mechanism
w_spc = utils.spacing_gen(10, -1, +1, dim=1)
#A,B,C,L,P = sig.params_init(10,mode='linear')
params = sig.params_init(num_sig=50,mode='random')


z_spc = sig.reparam(w_spc,params, indep=False)
dzdw = sig.df_dw(w_spc, params)
#z_spc = sig.reparam(w_spc, A,B,C,L,P, indep=True)



#disply.line_2d(w_spc, z_spc )



SAMPLING_SIZE = 1000

log_qw, w_gen = utils.uniform_init(-1,+1,dim=1)
log_pz, pz_gen = utils.gaussian_mix_init(np.array([1.0,-0.5,-2.]),np.array([0.1,0.2,0.05]),np.array([0.3,0.3,0.4]))

#sig.plot_qz(params,log_qw,target=log_qw, testing=True)
#w_samples = w_gen(SAMPLING_SIZE)

grad_kl = sig.grad_kl_init(log_pz, log_qw, params, w_gen, SAMPLING_SIZE)


trained_params = adam(grad_kl, params, step_size=0.1,num_iters=500)

#sig.plot_qz(params,log_qw,target=log_pz, testing=True)
sig.plot_qz(trained_params,log_qw,target=log_pz, testing=True)
#grad_A = sig.grad_kl(w_samples,log_pz, log_qw, params)
#print
#print grad_A
print('Done')