
# from wfpt import wiener_like_nn_weibull
# from wfpt import wiener_like_nn_angle 
# from wfpt import wiener_like_nn_ddm
# from wfpt import wiener_like_nn_ddm_analytic
# from wfpt import wiener_like_nn_levy
# from wfpt import wiener_like_nn_ornstein
# from wfpt import wiener_like_nn_ddm_sdv
# from wfpt import wiener_like_nn_ddm_sdv_analytic
# from wfpt import wiener_like_nn_full_ddm

import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist

#import wfpt

# Defining the likelihood functions
def make_cnn_likelihood(model):
    if model == 'ddm': # or model == 'weibull':
        def wienernn_like_ddm(x, 
                              v,
                              a,
                              z,
                              t,
                              p_outlier = 0,
                              w_outlier = 0,
                              **kwargs): #theta

            return hddm.wfpt.wiener_like_cnn_ddm(x['rt'],
                                                 x['response'], 
                                                 np.array([v, a, z, t], dtype = np.float32), 
                                                 p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                 w_outlier = w_outlier,
                                                 **kwargs)
        return wienernn_like_ddm
    else: 
        return 'Error model not implemented'