from exp1 import kl_div
import models as model
import utils_plots as disply
import matplotlib.pyplot as plt
import numpy as np


MAX_ITER = 500
# 2-dim example
DIM_Z = 2  # dimension of z

## defining target dist, p(z)
PROBS = np.array( [0.7, 0.3] )
MU_ARR  = np.array([ [1, 1],\
                   [-1, -1]  ])
VAR_ARR = np.zeros((2,2,2))
VAR_ARR[0,:,:] = np.eye(2)
VAR_ARR[1,:,:] = np.eye(2)/2.
log_pz,_ = model.gaussian_mix_init(MU_ARR, VAR_ARR, PROBS)

## defining the variational model
PSEUDO_SIZE = 1000
SAMPLING_SIZE = 100

gp_var = 1
gp_len = 100000
(gp_s,gp_t) = model.pseudo_data_gen(DIM_Z,PSEUDO_SIZE)

log_qw, w_gen = model.uniform_init(-1,+1)

w_samples = w_gen(SAMPLING_SIZE)

z = model.reparam(w_samples, gp_var, gp_len, gp_s, gp_t)
#plt.plot(gp_s, gp_t[:,1],'r--')
#plt.plot(w_samples, z[:,1], 'b.')

kl_distance = kl_div( gp_t, gp_var, gp_len, w_samples, gp_s, log_pz, log_qw )
print('KL: %s' % (kl_distance))

i=0
while (not np.isnan(kl_distance)) and (i<MAX_ITER) :
    i+=1
    w_samples = w_gen(SAMPLING_SIZE)
    kl_distance = kl_div( gp_t, gp_var, gp_len, w_samples, gp_s, log_pz, log_qw )
    print('KL -%s : %s' % (i,kl_distance))



#plt.show()