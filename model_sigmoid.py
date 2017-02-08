from models import VariationalModel

class SigmoidVI(VariationalModel):
    def __init__(self, num_sig, range_low, range_high):
        self.num_sig = num_sig

    def reparam(self, w):
        '''
        mapping w -> z, K->K
        :param w: N x K
        :return: N x K
        '''
        if w.ndim==1:
            w_ = w.reshape((1,len(w)))
            (w_.T - self.B)*self.L




