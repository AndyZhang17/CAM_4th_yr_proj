import matplotlib.pyplot as plt
import utils_plots as disply
import autograd.numpy as np
import utils as utils
import func_sigmoid as sig




# Testing the mapping mechanism
w_spc = utils.spacing_gen(10, -1, +1, dim=1)
A,B,C,L,P = sig.params_init(10,mode='linear')


z_spc = sig.reparam(w_spc, A,B,C,L,P, indep=False)
dzdw = sig.df_dw(w_spc, A,B,C,L,P)
#z_spc = sig.reparam(w_spc, A,B,C,L,P, indep=True)

print np.sum(P)
print dzdw



#disply.line_2d(w_spc, z_spc )



SAMPLING_SIZE = 1000

log_qw, w_gen = utils.uniform_init(-1,+1,dim=1)
log_pz, pz_gen = utils.gaussian_mix_init(np.array([0.5,-0.1]),np.array([0.5,0.3]),np.array([0.7,0.3]))

#sig.plot_qz(A,B,C,L,P,log_qw,target=log_qw, testing=False)
w_samples = w_gen(SAMPLING_SIZE)

grad_A = sig.grad_kl(w_samples,log_pz, log_qw, A,B,C,L,P)
print
print np.shape(grad_A)
print grad_A