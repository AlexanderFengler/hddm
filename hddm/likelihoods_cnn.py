
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

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values, 
                                               np.array([v, a, z, t], dtype = np.float32), 
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        return wienernn_like_ddm

    if model == 'weibull_cdf' or model == 'weibull':
        def wienernn_like_weibull(x, 
                                  v,
                                  a, 
                                  alpha,
                                  beta,
                                  z,
                                  t,
                                  p_outlier = 0,
                                  w_outlier = 0,
                                  **kwargs): #theta

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, alpha, beta], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        return wienernn_like_weibull

    if model == 'levy':
        def wienernn_like_levy(x, 
                               v, 
                               a, 
                               alpha,
                               z,
                               t,
                               p_outlier = 0.1,
                               w_outlier = 0.1,
                               **kwargs): #theta

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, alpha, t], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        return wienernn_like_levy

    if model == 'ornstein':
        def wienernn_like_ornstein(x, 
                                   v, 
                                   a, 
                                   g,
                                   z, 
                                   t,
                                   p_outlier = 0,
                                   w_outlier = 0,
                                   **kwargs): #theta
    
            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values, 
                                               np.array([v, a, z, g, t], dtype = np.float32),
                                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                               w_outlier = w_outlier,
                                               **kwargs)
        return wienernn_like_ornstein

    if model == 'full_ddm' or model == 'full_ddm2':
        def wienernn_like_full_ddm(x, 
                                   v, 
                                   sv, 
                                   a, 
                                   z, 
                                   sz, 
                                   t, 
                                   st, 
                                   p_outlier = 0,
                                   w_outlier = 0,
                                   **kwargs):

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, sz, sv, st], dtype = np.float32),
                                               p_outlier = p_outlier,
                                               w_outlier = w_outlier,
                                               **kwargs)

        return wienernn_like_full_ddm

    if model == 'angle':
        def wienernn_like_angle(x, 
                                v, 
                                a, 
                                theta, 
                                z,
                                t,
                                p_outlier = 0,
                                w_outlier = 0,
                                **kwargs):

            return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
                                               x['response'].values,
                                               np.array([v, a, z, t, theta], dtype = np.float32),
                                               p_outlier = p_outlier,
                                               w_outlier = w_outlier,
                                               **kwargs)

        return wienernn_like_angle
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'