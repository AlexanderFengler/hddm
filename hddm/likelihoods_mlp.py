
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

            def random(self):
                return partial(simulator, model = model, n_samples = self.shape, max_t = 20) # This may still be buggy !

            def pdf(self, x):
                print(self.parents)
                print(kwargs)
                return hddm.wfpt.wiener_like_nn_ddm_pdf(x, **self.parents, **kwargs) # This may still be buggy !

            def cdf(self, x):
                # TODO: Implement the CDF method for neural networks
                return 'Not yet implemented'

            # Create wfpt class
            wfpt_nn = stochastic_from_dist('Wienernn_' + model, partial(wienernn_like_ddm, **kwargs))

            wfpt_nn.pdf = pdf
            wfpt_nn.cdf_vec = None
            wfpt_nn.cdf = cdf
            wfpt_nn.random = random
            return wfpt_nn
    if model != 'ddm':
        return 'Not yet implemented for models other than the DDM'

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
        def wienernn_like_ddm_analytic(x, 
                                       v, 
                                       a, 
                                       z, 
                                       t,
                                       p_outlier = 0,
                                       w_outlier = 0,
                                       **kwargs):

            return hddm.wfpt.wiener_like_nn_ddm_analytic(x['rt'].values,
                                                         x['response'].values,  
                                                         v,
                                                         a,
                                                         z,
                                                         t, 
                                                         p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                                         w_outlier = w_outlier,
                                                         **kwargs)
        return wienernn_like_ddm_analytic

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

