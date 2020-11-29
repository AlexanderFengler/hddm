
"""
"""

from collections import OrderedDict
from copy import copy
import numpy as np
import pymc
import wfpt
import pickle

from kabuki.hierarchical import Knode # LOOK INTO KABUKI TO FIGURE OUT WHAT KNODE EXACTLY DOES
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from hddm.keras_models import load_mlp

from wfpt import wiener_like_nn_weibull
from wfpt import wiener_like_nn_angle 
from wfpt import wiener_like_nn_ddm
from wfpt import wiener_like_nn_ddm_analytic
from wfpt import wiener_like_nn_levy
from wfpt import wiener_like_nn_ornstein
from wfpt import wiener_like_nn_ddm_sdv
from wfpt import wiener_like_nn_ddm_sdv_analytic
from wfpt import wiener_like_nn_full_ddm

class HDDMnn(HDDM):
    """ HDDM model class that uses neural net likelihoods for
    WEIBULL, ANGLE,  ORNSTEIN, LEVY models.
    """

    def __init__(self, *args, **kwargs):
        kwargs['nn'] = True
        self.network_type = kwargs.pop('network_type', 'mlp')
        self.network = None #LAX
        self.non_centered = kwargs.pop('non_centered', False)
        self.w_outlier = kwargs.pop('w_outlier', 0.1)

        # Make model specific likelihood
        self.model = kwargs.pop('model', 'weibull')
        if self.model == 'ddm':
            #self.mlp = load_mlp(model = self.model)
            print('successfully loaded mlp')
            self.wfpt_nn = generate_wfpt_stochastic_class()
            #self.wfpt_nn = stochastic_from_dist('Wienernn_ddm', wienernn_like_ddm)
            #if self.network_type == 'MLP':
                #self.network = keras.models.load_model(...)

        if self.model == 'ddm_sdv':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_sdv', wienernn_like_ddm_sdv)
        
        if self.model == 'ddm_analytic':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_analytic', wienernn_like_ddm_analytic)

        if self.model == 'ddm_sdv_analytic':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_sdv_analytic', wienernn_like_ddm_sdv_analytic)
        
        if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull_cdf_concave':
            self.wfpt_nn = stochastic_from_dist('Wienernn_weibull', wienernn_like_weibull)
        
        if self.model == 'angle':
            self.wfpt_nn = stochastic_from_dist('Wienernn_angle', wienernn_like_angle) 

        if self.model == 'levy':
            self.wfpt_nn = stochastic_from_dist('Wienernn_levy', wienernn_like_levy) 

        if self.model == 'ornstein':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ornstein', wienernn_like_ornstein)

        if self.model == 'full_ddm' or self.model == 'full_ddm2':
            self.wfpt_nn = stochastic_from_dist('Wienernn_full_ddm', wienernn_like_full_ddm)
        super(HDDMnn, self).__init__(*args, **kwargs)
        print(self.p_outlier)
    
    def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()
        
        # SPLIT BY MODEL TO ACCOMMODATE TRAINED PARAMETER BOUNDS BY MODEL

        # PARAMETERS COMMON TO ALL MODELS
        if 'p_outlier' in include:
            knodes.update(self._create_family_invlogit('p_outlier',
                                                        value = 0.2,
                                                        g_tau = 10**-2,
                                                        std_std = 0.5
                                                        ))

        if self.model == 'weibull' or self.model == 'weibull_cdf':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.5,
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 2.5,
                                                               upper = 2.5,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           )) # should have lower = 0.2, upper = 0.8
            if 'alpha' in include:
                knodes.update(self._create_family_trunc_normal('alpha',
                                                               lower = 0.31, 
                                                               upper = 4.99, 
                                                               value = 2.34,
                                                               std_upper = 2
                                                               ))
            if 'beta' in include:
                knodes.update(self._create_family_trunc_normal('beta', 
                                                               lower = 0.31, 
                                                               upper = 6.99, 
                                                               value = 3.34,
                                                               std_upper = 2
                                                               ))

        if self.model == 'weibull_cdf_concave':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.5,
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 2.5,
                                                               upper = 2.5,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           )) # should have lower = 0.2, upper = 0.8
            if 'alpha' in include:
                knodes.update(self._create_family_trunc_normal('alpha',
                                                               lower = 1.00, # this guarantees initial concavity of the likelihood
                                                               upper = 4.99, 
                                                               value = 2.34,
                                                               std_upper = 2
                                                               ))
            if 'beta' in include:
                knodes.update(self._create_family_trunc_normal('beta', 
                                                               lower = 0.31, 
                                                               upper = 6.99, 
                                                               value = 3.34,
                                                               std_upper = 2
                                                               ))

        if self.model == 'ddm' or self.model == 'ddm_analytic':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.5,
                                                               value = 1.4,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 3.0,
                                                               upper = 3.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           )) # should have lower = 0.1, upper = 0.9

            print(knodes.keys())

        
        if self.model == 'ddm_sdv' or self.model == 'ddm_sdv_analytic':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.5,
                                                               value = 1.4,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 3.0,
                                                               upper = 3.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           )) # should have lower = 0.1, upper = 0.9
            if 'sv' in include:
                knodes.update(self._create_family_trunc_normal('sv', 
                                                               lower = 1e-3,
                                                               upper = 2.5, 
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))

        if self.model == 'full_ddm' or self.model == 'full_ddm2':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.5,
                                                               value = 1.4,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 3.0,
                                                               upper = 3.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 0.25,
                                                               upper = 2.25, 
                                                               value = .5,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           )) # should have lower = 0.1, upper = 0.9

            if 'sz' in include:
                knodes.update(self._create_family_trunc_normal('sz', 
                                                               lower = 1e-3,
                                                               upper = 0.2, 
                                                               value = 0.1,
                                                               std_upper = 0.1 # added AF
                                                               ))

            if 'sv' in include:
                knodes.update(self._create_family_trunc_normal('sv', 
                                                               lower = 1e-3,
                                                               upper = 2.0, 
                                                               value = 1.0,
                                                               std_upper = 0.5 # added AF
                                                               ))

            if 'st' in include:
                knodes.update(self._create_family_trunc_normal('st', 
                                                               lower = 1e-3,
                                                               upper = 0.25, 
                                                               value = 0.125,
                                                               std_upper = 0.1 # added AF
                                                               ))

        if self.model == 'angle':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.0,
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 3.0,
                                                               upper = 3.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           ))
            if 'theta' in include:
                knodes.update(self._create_family_trunc_normal('theta',
                                                               lower = -0.1, 
                                                               upper = 1.45, 
                                                               value = 0.5,
                                                               std_upper = 1
                                                               )) # should have lower = 0.2, upper = 0.8

        if self.model == 'ornstein':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.0,
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 2.0,
                                                               upper = 2.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           ))
            if 'g' in include:
                knodes.update(self._create_family_trunc_normal('g',
                                                               lower = -1.0, 
                                                               upper = 1.0, 
                                                               value = 0.5,
                                                               std_upper = 1
                                                               )) # should have lower = 0.2, upper = 0.8
        
        if self.model == 'levy':
            if 'a' in include:
                knodes.update(self._create_family_trunc_normal('a',
                                                               lower = 0.3,
                                                               upper = 2.0,
                                                               value = 1,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'v' in include:
                knodes.update(self._create_family_trunc_normal('v', 
                                                               lower = - 3.0,
                                                               upper = 3.0,
                                                               value = 0,
                                                               std_upper = 1.5
                                                               ))
            if 't' in include:
                knodes.update(self._create_family_trunc_normal('t', 
                                                               lower = 1e-3,
                                                               upper = 2, 
                                                               value = .01,
                                                               std_upper = 1 # added AF
                                                               ))
            if 'z' in include:
                knodes.update(self._create_family_invlogit('z',
                                                           value = .5,
                                                           g_tau = 10**-2,
                                                           std_std = 0.5
                                                           ))
            if 'alpha' in include:
                knodes.update(self._create_family_trunc_normal('alpha',
                                                               lower = 1.0, 
                                                               upper = 2.0, 
                                                               value = 1.5,
                                                               std_upper = 1
                                                               ))
                                                               # should have lower = 0.1, upper = 0.9
                      
        print('knodes')
        print(knodes)

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5
        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier
        wfpt_parents['w_outlier'] = self.w_outlier # likelihood of an outlier point
        
        # MODEL SPECIFIC PARAMETERS
        if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull_cdf_concave':
            wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 3 
            wfpt_parents['beta'] = knodes['beta_bottom'] if 'beta' in self.include else 3
        
        if self.model == 'ornstein':
            wfpt_parents['g'] = knodes['g_bottom'] if 'g' in self.include else 0
        
        if self.model == 'levy':
            wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 2
        
        if self.model == 'angle':
            wfpt_parents['theta'] = knodes['theta_bottom'] if 'theta' in self.include else 0

        if self.model == 'full_ddm' or self.model =='full_ddm2':
            wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0 #self.default_intervars['sv']
            wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0 #self.default_intervars['sz']
            wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0 #self.default_intervars['st']

        print('wfpt parents: ')
        print(wfpt_parents)
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        # my_dict = {'wfpt_parents':wfpt_parents, 'network':self.network} #LAX

        return Knode(self.wfpt_nn, 
                     'wfpt', 
                     observed = True, 
                     col_name = ['response', 'rt'], # TODO: One could preprocess at initialization
                     **wfpt_parents)
        #return Knode(..., **my_dict)

def generate_wfpt_stochastic_class(wiener_params = None, model = 'ddm'):
    """
    create a wfpt stochastic class by creating a pymc node and then adding quantile functions.
    Input:
        wiener_params <dict> - dictonary of wiener_params for wfpt 
        sampling_method <string> - an argument used by hddm.generate.gen_rts
        cdf_range <sequence> -  an argument used by hddm.generate.gen_rts
        sampling_dt <float> - an argument used by hddm.generate.gen_rts
    Ouput:
        wfpt <class> - the wfpt stochastic
    """
    
    mlp = load_mlp(model = model)

    def wfpt_like(x, v, sv, a, z, t, p_outlier = 0, w_outlier = 0):
        size = x.shape[0]
        n_params = 4
        ll_min = -16.11809
        data = np.zeros((size, n_params, + 2), dtype = np.float32)
        data[:, :n_params] = np.tile([v,a,z,t], (size, 1))
        data[:, n_params:] = x.values
        return np.sum(np.core.umath.maximum(mlp.predict_on_batch(data), ll_min))
   
    wfpt = stochastic_from_dist('wfpt', wfpt_like)

    #wfpt.pdf = pdf
    #wfpt.cdf_vec = lambda self: hddm.wfpt.gen_cdf_using_pdf(time=cdf_range[1], **dict(list(self.parents.items()) + list(wp.items())))
    #wfpt.cdf = cdf
    #wfpt.random = random

    #add quantiles functions
    #add_quantiles_functions_to_pymc_class(wfpt)
    return wfpt

def wienernn_like_weibull(x, 
                          v,
                          a, 
                          alpha,
                          beta,
                          z,
                          t,
                          p_outlier = 0,
                          w_outlier = 0): #theta

    return wiener_like_nn_weibull(x['rt'].values,
                                  x['response'].values, 
                                  v, 
                                  a, 
                                  alpha, 
                                  beta,
                                  z, 
                                  t, 
                                  p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                  w_outlier = w_outlier)

def wienernn_like_ddm(x, 
                      v,  
                      a, 
                      z,  
                      t, 
                      p_outlier = 0,
                      w_outlier = 0.1):

    return wiener_like_nn_ddm(x['rt'].values,
                              x['response'].values,  
                              v, # sv,
                              a, 
                              z, # sz,
                              t, # st,
                              p_outlier = p_outlier,
                              w_outlier = w_outlier)


def wienernn_like_levy(x, 
                       v, 
                       a, 
                       alpha,
                       z,
                       t,
                       p_outlier = 0.1,
                       w_outlier = 0.1): #theta

    return wiener_like_nn_levy(x['rt'].values,
                               x['response'].values, 
                               v,
                               a, 
                               alpha, 
                               z,
                               t, 
                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                               w_outlier = w_outlier)

def wienernn_like_ornstein(x, 
                           v, 
                           a, 
                           g,
                           z, 
                           t,
                           p_outlier = 0,
                           w_outlier = 0): #theta
    
    return wiener_like_nn_ornstein(x['rt'].values,
                                   x['response'].values, 
                                   v, 
                                   a, 
                                   g, 
                                   z, 
                                   t, 
                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                   w_outlier = w_outlier)

def wienernn_like_ddm_analytic(x, 
                               v, 
                               a, 
                               z, 
                               t,
                               p_outlier = 0,
                               w_outlier = 0):

    return wiener_like_nn_ddm_analytic(x['rt'].values,
                                       x['response'].values,  
                                       v,
                                       a,
                                       z,
                                       t, 
                                       p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                       w_outlier = w_outlier)

def wienernn_like_ddm_sdv(x, 
                          v,
                          sv,
                          a, 
                          z, 
                          t,
                          p_outlier = 0,
                          w_outlier = 0):

    return wiener_like_nn_ddm_sdv(x['rt'].values,
                                  x['response'].values,  
                                  v,
                                  sv,
                                  a,
                                  z, 
                                  t, 
                                  p_outlier = p_outlier,
                                  w_outlier = w_outlier)

def wienernn_like_ddm_sdv_analytic(x, 
                                   v, 
                                   sv,
                                   a, 
                                   z, 
                                   t, 
                                   p_outlier = 0,
                                   w_outlier = 0):

    return wiener_like_nn_ddm_sdv_analytic(x['rt'].values,
                                           x['response'].values,  
                                           v, 
                                           sv, 
                                           a, 
                                           z, 
                                           t,
                                           p_outlier = p_outlier,
                                           w_outlier = w_outlier)

def wienernn_like_full_ddm(x, 
                           v, 
                           sv, 
                           a, 
                           z, 
                           sz, 
                           t, 
                           st, 
                           p_outlier = 0,
                           w_outlier = 0):

    return wiener_like_nn_full_ddm(x['rt'].values,
                                   x['response'].values,  
                                   v, 
                                   sv, 
                                   a, 
                                   z, 
                                   sz, 
                                   t, 
                                   st, 
                                   p_outlier = p_outlier,
                                   w_outlier = w_outlier)

def wienernn_like_angle(x, 
                        v, 
                        a, 
                        theta, 
                        z,
                        t,
                        p_outlier = 0,
                        w_outlier = 0):

    return wiener_like_nn_angle(x['rt'].values,
                                x['response'].values,  
                                v,
                                a, 
                                theta,
                                z,
                                t,
                                p_outlier = p_outlier,
                                w_outlier = w_outlier)