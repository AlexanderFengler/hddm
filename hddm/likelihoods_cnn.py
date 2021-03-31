
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
from hddm.simulators import *
import data_simulators


#import wfpt


# Defining the likelihood functions
def make_cnn_likelihood(model, pdf_multiplier = 1,  **kwargs):
    def random(self):
        # print(self.parents)
        # print('printing the dir of self.parents directly')
        # print(dir(self.parents))
        # print('printing dir of the v variable')
        # print(dir(self.parents['v']))
        # print(self.parents['v'].value)
        # print(self.parents.value)
        # print('trying to print the value part of parents')
        # print(dict(self.parents.value))
        # print('tying to pring the values part of parents')
        # print(self.parents.values)

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]['default_params'], dtype = np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0
        
        for param in model_config[model]['params']:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1
        
        print('print theta from random function in wfpt_nn')
        print(theta)

        #new_func = partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !
        #print('self shape: ')
        #print(self.shape)
        sim_out = simulator(theta = theta, model = model, n_samples = self.shape[0], max_t = 20)
        return hddm_preprocess(sim_out)

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

        def pdf_ddm(self, x):
            #print('type of x')
            #print(type(x))
            #print(x)
            #print(self.parents)
            #print(**self.parents)
            #print(self.parents['a'])
            #print(dir(self.parents['a']))
            #print(self.parents['a'].value)
            #print(kwargs)
            #print(self.parents['a'].value)
            # Note as per kabuki it seems that x tends to come in as a 'value_range', which is essetially a 1d ndarray
            # We could change this ...

            #rt = np.array()
            rt = np.array(x, dtype = np.int_)
            response = rt
            response[rt < 0] = 0
            response[rt > 0] = 1
            response = response.astype(np.int_)

            #response = rt / np.abs(rt)
            #rt = np.abs(rt)
           
            # this can be simplified so that we pass parameters directly to the simulator ...
            theta = np.array(model_config[model]['default_params'], dtype = np.float32)
            keys_tmp = self.parents.value.keys()
            cnt = 0
            
            for param in model_config[model]['params']:
                if param in keys_tmp:
                    theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
                cnt += 1

            #print(rt)
            #print(response)
            #print(response.shape)
            #print(rt.shape)
            # response = 
            #pdf_fun = hddm.wfpt.wiener_like_nn_ddm_pdf
            # model_config[] # TODO FILL THIS IN SO THAT WE CREATE THE APPROPRIATE ARRAY AS INPUT TO THE SIMULATOR
            out = pdf_multiplier * hddm.wfpt.wiener_pdf_cnn_2(x = rt, response = response, network = kwargs['network'], parameters = theta)# **kwargs) # This may still be buggy !
            return out

        
        def cdf_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm, **kwargs))

        wfpt_nn.pdf = pdf_ddm
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm
        wfpt_nn.random = random
        return wfpt_nn

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
                               p_outlier = 0,
                               w_outlier = 0,
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


# def make_cnn_likelihood(model):
#     if model == 'ddm': # or model == 'weibull':
#         def wienernn_like_ddm(x, 
#                               v,
#                               a,
#                               z,
#                               t,
#                               p_outlier = 0,
#                               w_outlier = 0,
#                               **kwargs): #theta

#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values, 
#                                                np.array([v, a, z, t], dtype = np.float32), 
#                                                p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
#                                                w_outlier = w_outlier,
#                                                **kwargs)
#         return wienernn_like_ddm

#     if model == 'weibull_cdf' or model == 'weibull':
#         def wienernn_like_weibull(x, 
#                                   v,
#                                   a, 
#                                   alpha,
#                                   beta,
#                                   z,
#                                   t,
#                                   p_outlier = 0,
#                                   w_outlier = 0,
#                                   **kwargs): #theta

#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values,
#                                                np.array([v, a, z, t, alpha, beta], dtype = np.float32),
#                                                p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
#                                                w_outlier = w_outlier,
#                                                **kwargs)
#         return wienernn_like_weibull

#     if model == 'levy':
#         def wienernn_like_levy(x, 
#                                v, 
#                                a, 
#                                alpha,
#                                z,
#                                t,
#                                p_outlier = 0,
#                                w_outlier = 0,
#                                **kwargs): #theta

#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values,
#                                                np.array([v, a, z, alpha, t], dtype = np.float32),
#                                                p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
#                                                w_outlier = w_outlier,
#                                                **kwargs)
#         return wienernn_like_levy

#     if model == 'ornstein':
#         def wienernn_like_ornstein(x, 
#                                    v, 
#                                    a, 
#                                    g,
#                                    z, 
#                                    t,
#                                    p_outlier = 0,
#                                    w_outlier = 0,
#                                    **kwargs): #theta
    
#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values, 
#                                                np.array([v, a, z, g, t], dtype = np.float32),
#                                                p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
#                                                w_outlier = w_outlier,
#                                                **kwargs)
#         return wienernn_like_ornstein

#     if model == 'full_ddm' or model == 'full_ddm2':
#         def wienernn_like_full_ddm(x, 
#                                    v, 
#                                    sv, 
#                                    a, 
#                                    z, 
#                                    sz, 
#                                    t, 
#                                    st, 
#                                    p_outlier = 0,
#                                    w_outlier = 0,
#                                    **kwargs):

#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values,
#                                                np.array([v, a, z, t, sz, sv, st], dtype = np.float32),
#                                                p_outlier = p_outlier,
#                                                w_outlier = w_outlier,
#                                                **kwargs)

#         return wienernn_like_full_ddm

#     if model == 'angle':
#         def wienernn_like_angle(x, 
#                                 v, 
#                                 a, 
#                                 theta, 
#                                 z,
#                                 t,
#                                 p_outlier = 0,
#                                 w_outlier = 0,
#                                 **kwargs):

#             return hddm.wfpt.wiener_like_cnn_2(x['rt'].values,
#                                                x['response'].values,
#                                                np.array([v, a, z, t, theta], dtype = np.float32),
#                                                p_outlier = p_outlier,
#                                                w_outlier = w_outlier,
#                                                **kwargs)

#         return wienernn_like_angle
#     else:
#         return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'