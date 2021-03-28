
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

def make_mlp_likelihood_complete(model, **kwargs):
    if model == 'ddm':
        def wienernn_like_ddm(x, 
                                v,  
                                a, 
                                z,  
                                t, 
                                p_outlier = 0,
                                w_outlier = 0.1,
                                **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm(x['rt'].values,
                                                x['response'].values,  
                                                v, # sv,
                                                a, 
                                                z, # sz,
                                                t, # st,
                                                p_outlier = p_outlier,
                                                w_outlier = w_outlier,
                                                **kwargs)

        def random_ddm(self):
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
            print('self shape: ')
            print(self.shape)
            sim_out = simulator(theta = theta, model = model, n_samples = self.shape[0], max_t = 20)
            return hddm_preprocess(sim_out)

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

            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            
            #print(rt)
            #print(response)
            #print(response.shape)
            #print(rt.shape)
            # response = 
            #pdf_fun = hddm.wfpt.wiener_like_nn_ddm_pdf
            # model_config[] # TODO FILL THIS IN SO THAT WE CREATE THE APPROPRIATE ARRAY AS INPUT TO THE SIMULATOR
            out = hddm.wfpt.wiener_like_nn_ddm_pdf(x = rt, response = response, network = kwargs['network'], **self.parents)# **kwargs) # This may still be buggy !
            return out

        def cdf_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm, **kwargs))

        wfpt_nn.pdf = pdf_ddm
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm
        wfpt_nn.random = random_ddm
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

            return hddm.wfpt.wiener_like_nn_weibull(x['rt'].values,
                                                    x['response'].values, 
                                                    v, 
                                                    a, 
                                                    alpha, 
                                                    beta,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        def random_weibull(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_weibull(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_weibull_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_weibull(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_weibull, **kwargs))

        wfpt_nn.pdf = pdf_weibull
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_weibull
        wfpt_nn.random = random_weibull
        return wfpt_nn
    
    if model == 'ddm_sdv':
        def wienernn_like_ddm_sdv(x, 
                          v,
                          sv,
                          a, 
                          z, 
                          t,
                          p_outlier = 0,
                          w_outlier = 0,
                          **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm_sdv(x['rt'].values,
                                                    x['response'].values,  
                                                    v,
                                                    sv,
                                                    a,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier,
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        def random_ddm_sdv(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_ddm_sdv(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ddm_sdv_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ddm_sdv(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm_sdv, **kwargs))

        wfpt_nn.pdf = pdf_ddm_sdv
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm_sdv
        wfpt_nn.random = random_ddm_sdv
        return wfpt_nn
    
    if model == 'ddm_sdv_analytic':
        def wienernn_like_ddm_sdv_analytic(x, 
                                           v, 
                                           sv,
                                           a, 
                                           z, 
                                           t, 
                                           p_outlier = 0,
                                           w_outlier = 0,
                                           **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm_sdv_analytic(x['rt'].values,
                                                             x['response'].values,  
                                                             v, 
                                                             sv, 
                                                             a, 
                                                             z, 
                                                             t,
                                                             p_outlier = p_outlier,
                                                             w_outlier = w_outlier,
                                                             **kwargs)

        def random_ddm_sdv_analytic(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_ddm_sdv_analytic(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ddm_sdv_analytic_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ddm_sdv_analytic(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm_sdv_analytic, **kwargs))

        wfpt_nn.pdf = pdf_ddm_sdv_analytic
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ddm_sdv_analytic
        wfpt_nn.random = random_ddm_sdv_analytic
        #return wienernn_like_ddm_sdv_analytic
        return wfpt_nn

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

            return hddm.wfpt.wiener_like_nn_levy(x['rt'].values,
                                    x['response'].values, 
                                    v,
                                    a, 
                                    alpha, 
                                    z,
                                    t, 
                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                    w_outlier = w_outlier,
                                    **kwargs)

        def random_levy(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_levy(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_levy_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_levy(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_levy, **kwargs))

        wfpt_nn.pdf = pdf_levy
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_levy
        wfpt_nn.random = random_levy
        #return wienernn_like_ddm_sdv_analytic
        return wfpt_nn

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
    
            return hddm.wfpt.wiener_like_nn_ornstein(x['rt'].values,
                                                     x['response'].values, 
                                                     v, 
                                                     a, 
                                                     g, 
                                                     z, 
                                                     t, 
                                                     p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                     w_outlier = w_outlier,
                                                     **kwargs)
        #return wienernn_like_ornstein

        def random_ornstein(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_ornstein(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_ornstein_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_ornstein(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ornstein, **kwargs))

        wfpt_nn.pdf = pdf_ornstein
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_ornstein
        wfpt_nn.random = random_ornstein
        #return wienernn_like_ddm_sdv_analytic
        #return wienernn_like_ornstein
        return wfpt_nn

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

            return hddm.wfpt.wiener_like_nn_full_ddm(x['rt'].values,
                                                     x['response'].values,
                                                     v,
                                                     sv,
                                                     a,
                                                     z, 
                                                     sz, 
                                                     t,
                                                     st,
                                                     p_outlier = p_outlier,
                                                     w_outlier = w_outlier,
                                                     **kwargs)

        #return wienernn_like_full_ddm
        def random_full_ddm(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_full_ddm(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_full_ddm_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_full_ddm(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_full_ddm, **kwargs))

        wfpt_nn.pdf = pdf_full_ddm
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_full_ddm
        wfpt_nn.random = random_full_ddm
        #return wienernn_like_ddm_sdv_analytic
        #return wienernn_like_ornstein
        return wfpt_nn

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

            return hddm.wfpt.wiener_like_nn_angle(x['rt'].values,
                                                  x['response'].values,  
                                                  v,
                                                  a, 
                                                  theta,
                                                  z,
                                                  t,
                                                  p_outlier = p_outlier,
                                                  w_outlier = w_outlier,
                                                  **kwargs)
            
        #return wienernn_like_full_ddm
        def random_angle(self):
            return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

        def pdf_angle(self, x):
            rt = np.array(x, dtype = np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            out = hddm.wfpt.wiener_like_nn_angle_pdf(x = rt, response = response, network = kwargs['network'], **self.parents) # **kwargs) # This may still be buggy !
            return out

        def cdf_angle(self, x):
            # TODO: Implement the CDF method for neural networks
            return 'Not yet implemented'

        # Create wfpt class
        wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_angle, **kwargs))

        wfpt_nn.pdf = pdf_angle
        wfpt_nn.cdf_vec = None # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_angle
        wfpt_nn.random = random_angle
        #return wienernn_like_ddm_sdv_analytic
        #return wienernn_like_ornstein
        
        #return wienernn_like_angle
        return wfpt_nn
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'
    #print('printing wfpt_nn')
    #print(wfpt_nn)
    #return wfpt_nn

    # # if model != 'ddm':
    # #     return 'Not yet implemented for models other than the DDM'

# Defining the likelihood functions
def make_mlp_likelihood(model):
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

            return hddm.wfpt.wiener_like_nn_weibull(x['rt'].values,
                                                    x['response'].values, 
                                                    v, 
                                                    a, 
                                                    alpha, 
                                                    beta,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                    w_outlier = w_outlier,
                                                    **kwargs)
        return wienernn_like_weibull

    if model == 'ddm':
        def wienernn_like_ddm(x, 
                              v,  
                              a, 
                              z,  
                              t, 
                              p_outlier = 0,
                              w_outlier = 0.1,
                              **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm(x['rt'].values,
                                                x['response'].values,  
                                                v, # sv,
                                                a, 
                                                z, # sz,
                                                t, # st,
                                                p_outlier = p_outlier,
                                                w_outlier = w_outlier,
                                                **kwargs)
        return wienernn_like_ddm

    if model == 'ddm_sdv':
        def wienernn_like_ddm_sdv(x, 
                          v,
                          sv,
                          a, 
                          z, 
                          t,
                          p_outlier = 0,
                          w_outlier = 0,
                          **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm_sdv(x['rt'].values,
                                                    x['response'].values,  
                                                    v,
                                                    sv,
                                                    a,
                                                    z, 
                                                    t, 
                                                    p_outlier = p_outlier,
                                                    w_outlier = w_outlier,
                                                    **kwargs)
        return wienernn_like_ddm_sdv

    if model == 'ddm_sdv_analytic':
        def wienernn_like_ddm_sdv_analytic(x, 
                                           v, 
                                           sv,
                                           a, 
                                           z, 
                                           t, 
                                           p_outlier = 0,
                                           w_outlier = 0,
                                           **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm_sdv_analytic(x['rt'].values,
                                                x['response'].values,  
                                                v, 
                                                sv, 
                                                a, 
                                                z, 
                                                t,
                                                p_outlier = p_outlier,
                                                w_outlier = w_outlier,
                                                **kwargs)
        return wienernn_like_ddm_sdv_analytic

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

            return hddm.wfpt.wiener_like_nn_levy(x['rt'].values,
                                    x['response'].values, 
                                    v,
                                    a, 
                                    alpha, 
                                    z,
                                    t, 
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
    
            return hddm.wfpt.wiener_like_nn_ornstein(x['rt'].values,
                                                     x['response'].values, 
                                                     v, 
                                                     a, 
                                                     g, 
                                                     z, 
                                                     t, 
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

            return hddm.wfpt.wiener_like_nn_full_ddm(x['rt'].values,
                                                     x['response'].values,
                                                     v,
                                                     sv,
                                                     a,
                                                     z, 
                                                     sz, 
                                                     t,
                                                     st,
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

            return hddm.wfpt.wiener_like_nn_angle(x['rt'].values,
                                                  x['response'].values,  
                                                  v,
                                                  a, 
                                                  theta,
                                                  z,
                                                  t,
                                                  p_outlier = p_outlier,
                                                  w_outlier = w_outlier,
                                                  **kwargs)
        
        return wienernn_like_angle
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'


# Defining only the model likelihood at this point !
def generate_wfpt_nn_ddm_reg_stochastic_class(wiener_params = None,
                                              sampling_method = 'cdf',
                                              cdf_range = (-5, 5), 
                                              sampling_dt = 1e-4,
                                              model = None,
                                              **kwargs):

    if model == 'ddm':
        def wiener_multi_like_nn_ddm(value, v, a, z, t, 
                                    reg_outcomes, 
                                    p_outlier = 0, 
                                    w_outlier = 0.1,
                                    **kwargs):

            """Log-likelihood for the full DDM using the interpolation method"""

            params = {'v': v, 'a': a, 'z': z, 't': t}
            n_params = int(4)
            size = int(value.shape[0])
            data = np.zeros((size, 6), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_ddm(data,
                                                      p_outlier = p_outlier,
                                                      w_outlier = w_outlier,
                                                      **kwargs)

        # Need to rewrite these random parts !
        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_ddm, **kwargs))
        stoch.random = random

    if model == 'full_ddm' or model == 'full_ddm2':
        def wiener_multi_like_nn_full_ddm(value, v, sv, a, z, sz, t, st, 
                                         reg_outcomes, 
                                         p_outlier = 0, 
                                         w_outlier = 0.1,
                                         **kwargs):

            params = {'v': v, 'a': a, 'z': z, 't': t, 'sz': sz, 'sv': sv, 'st': st}

            n_params = int(7)
            size = int(value.shape[0])
            data = np.zeros((size, 9), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'sz', 'sv', 'st']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_full_ddm(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_full_ddm, **kwargs))
        stoch.random = random

    if model == 'angle':
        def wiener_multi_like_nn_angle(value, v, a, theta, z, t, 
                                   reg_outcomes, 
                                   p_outlier = 0, 
                                   w_outlier = 0.1,
                                   **kwargs):

            """Log-likelihood for the full DDM using the interpolation method"""

            params = {'v': v, 'a': a, 'z': z, 't': t, 'theta': theta}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'theta']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_angle(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_angle, **kwargs))
        stoch.random = random

    if model == 'levy':
        def wiener_multi_like_nn_levy(value, v, a, alpha, z, t, 
                                        reg_outcomes, 
                                        p_outlier = 0, 
                                        w_outlier = 0.1,
                                        **kwargs):

            """Log-likelihood for the full DDM using the interpolation method"""

            params = {'v': v, 'a': a, 'z': z, 'alpha': alpha, 't': t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 'alpha', 't']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_levy(data,
                                                    p_outlier = p_outlier,
                                                    w_outlier = w_outlier,
                                                    **kwargs)

        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_levy, **kwargs))
        stoch.random = random
    
    if model == 'ornstein':
        def wiener_multi_like_nn_ornstein(value, v, a, g, z, t, 
                                      reg_outcomes, 
                                      p_outlier = 0, 
                                      w_outlier = 0.1,
                                      **kwargs):

            params = {'v': v, 'a': a, 'z': z, 'g': g, 't': t}
            
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, 7), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 'g', 't']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_ornstein(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)

        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_ornstein, **kwargs))
        stoch.random = random

    if model == 'weibull_cdf' or model == 'weibull':
        def wiener_multi_like_nn_weibull(value, v, a, alpha, beta, z, t, 
                                         reg_outcomes, 
                                         p_outlier = 0, 
                                         w_outlier = 0.1,
                                         **kwargs):

            params = {'v': v, 'a': a, 'z': z, 't': t, 'alpha': alpha, 'beta': beta}
            n_params = int(6)
            size = int(value.shape[0])
            data = np.zeros((size, 8), dtype = np.float32)
            data[:, n_params:] = np.stack([ np.absolute(value['rt']).astype(np.float32), value['response'].astype(np.float32) ], axis = 1)

            cnt = 0
            for tmp_str in ['v', 'a', 'z', 't', 'alpha', 'beta']:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # THIS IS NOT YET FINISHED !
            return hddm.wfpt.wiener_like_multi_nn_weibull(data,
                                                        p_outlier = p_outlier,
                                                        w_outlier = w_outlier,
                                                        **kwargs)


        def random(self):
            param_dict = deepcopy(self.parents.value)
            del param_dict['reg_outcomes']
            sampled_rts = self.value.copy()

            for i in self.value.index:
                #get current params
                for p in self.parents['reg_outcomes']:
                    param_dict[p] = np.asscalar(self.parents.value[p].loc[i])
                #sample
                samples = hddm.generate.gen_rts(method=sampling_method,
                                                size=1, dt=sampling_dt, **param_dict)

                sampled_rts.loc[i, 'rt'] = hddm.utils.flip_errors(samples).rt

            return sampled_rts

        stoch = stochastic_from_dist('wfpt_reg', partial(wiener_multi_like_nn_weibull, **kwargs))
        stoch.random = random
    return stoch

