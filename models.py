import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.numpy.linalg as linalg
from autograd import grad

PI = np.pi
INF = 1e+10

class VariationalModel(object):
    def __init__(self):
        pass

    def reparam(self):
        # mapping w->z
        pass

    def log_q_reparam(self):
        # w->z, q(z)
        pass
    def param_update(self):
        pass



def pseudo_data_gen( dim_t, gp_size, dim_s=1):
    '''
    Pseudodata uniformly between -1, +1
    :param dim_t:
    :param gp_size:
    :return:
    '''
    dim_s = dim_t
    s = np.linspace( -1.0, +1.0, num=gp_size )
    t = np.reshape( np.tile(s,dim_t), (dim_t, gp_size) ).T
    if dim_s!=1:
        s = np.reshape( np.tile(s,dim_s), (dim_s,gp_size)).T
    return s,t


def reparam_1d( w, var_ard, len_weight, s, t ):
    ''' mean-GP mapping: w->z'''
    # assuming dim_w, dim_s = 1
    s_ = s.reshape((1,len(s)))
    w_ = w.reshape((1,len(w)))
    s_sub_w = s_-w_.T
    Kws = var_ard * np.exp( np.power((s_sub_w),2)/-2.*len_weight )
    Kss = var_ard * np.exp( np.power((s_-s_.T),2)/-2.*len_weight )
    out = np.dot(np.dot(Kws,linalg.inv(Kss)),t)
    return out

def reparam(w, var_ard, len_weight,s,t):
    '''dim_s = dim_t'''
    if s.ndim==1:
        out = reparam_1d(w, var_ard, len_weight, s, t)
    else:
        out = np.zeros(np.shape(w))
        dim_s = np.shape(out)[1]
        for d in range(dim_s):
            out[:,d] = reparam_1d(w[:,d], var_ard,len_weight,s[:,d],t[:,d])
    return out

df_dw_1d = grad(reparam_1d, argnum=0)
df_dw = grad(reparam, argnum=0)

# def gp_diff(w,var_ard, len_weight, s,t):
#     return df_dw(w, var_ard, len_weight,s, t)

def gp_diff(w, var_ard, len_weight, s,t):
    ''' R^n->R^n mapping '''
    if w.ndim==1:
        out = df_dw_1d(w,var_ard, len_weight, s,t)
    else:
        N, dim_w = np.shape( w )
        out = np.zeros( (N, dim_w) )
        for d in range(dim_w):
            temp = df_dw_1d(w[:,d], var_ard, len_weight, s[:,d], t[:,d])
            out[:,d] = temp
    return out



#
# def gp_diff(w, var_ard,len_weight,s,t):
#     '''function df/dw, where f is a mapping w->z
#     R -> R^n
#     :param: w, N*1
#     return: df/dw, N*n
#     '''
#     s_ = s.reshape((1,len(s)))
#     w_ = w.reshape((1,len(w)))
#     s_sub_w = s_-w_.T
#     Kws = var_ard * np.exp( np.power((s_sub_w),2)/(-2.)*len_weight )
#     Kss = var_ard * np.exp( np.power((s_-s_.T),2)/(-2.)*len_weight )
#     Kws_diff = np.multiply( Kws, s_sub_w ) *len_weight
#     return np.dot(np.dot(Kws_diff,linalg.inv(Kss)),t)


def log_q_reparam(w, var_ard,len_weight, s,t, log_qw):
    df_dw = gp_diff(w,var_ard,len_weight,s,t)
    if t.ndim !=1:
        log_qz = np.sum(log_qw(w),1) - np.sum( np.log(np.abs(df_dw)) ,1 )
    else:
        log_qz = log_qw(w) - np.log(np.abs(df_dw))
    return log_qz

def mono_update(gp_t, grad_t, LEARNING_RATE):
    pass



