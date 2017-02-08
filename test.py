import models as exp
import utils_plots as disply
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import models as model
from autograd import elementwise_grad as ele_grad



# # dim-1 testing
# mu1 = 1
# var1 = 0.5
# p1, g1 = exp.gaussian_init(np.array([mu1]), np.array([var1]))
# x1 = g1(10000)
# y_ = np.exp( np.array( [p1(x) for x in x1] ) )
# #ax1 = disply.hist_1d(x1)
# #ax1.plot(x1,np.exp(p1(x1)),'r.')
#
#
# # dim-2 Gaussian mixture model
# mu2_0 = np.array([1,1])
# var2_0 = np.array([ [1, 0.4],\
# 				    [0.4, 1] ] )/8
# mu2_1 = np.array([-1,-1])
# var2_1 = np.eye(2)/5.
# probs = np.array([0.7,0.3])
#
# p_mix, g_mix = utils.gaussian_mix_init(np.array([mu2_0,mu2_1]),\
#                                      np.array([var2_0,var2_1]),probs)
# x3 = g_mix(20)
# print( np.exp( p_mix(x3)) )
# disply.hist_2d( x3[:,0], x3[:,1] )



####
PSEUDO_SIZE = 10
len_weight = 1
var_ard = 1
DIM = 1

log_prior, w_gen = utils.uniform_init(-1,+1,dim=DIM)

s,t = model.pseudo_data_gen( DIM, PSEUDO_SIZE,dim_s=DIM)
w_samples, _ = model.pseudo_data_gen(1, 10*PSEUDO_SIZE )
#w_samples = w_gen(500)

z = exp.reparam_1d(w_samples, var_ard, len_weight, s, t)
dd = ele_grad(exp.reparam_1d, argnum=0)
df_dw = dd( w_samples, var_ard, len_weight, s, t  )

#plt.plot(s, t[:,0], 'r--')
#plt.plot(w_samples, z, 'b.' )

print(np.shape(df_dw))

#plt.plot(w_samples, df_dw, 'r--')
disply.hist_1d(df_dw)
print(np.shape(df_dw))
print(np.mean(np.abs(df_dw)))
print(np.mean(df_dw))

plt.show()


