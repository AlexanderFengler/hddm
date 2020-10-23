
"""
"""

from collections import OrderedDict
from copy import copy
import numpy as np
import pymc
import wfpt
import pickle

from kabuki.hierarchical import Knode # LOOK INTO KABUKI TO FIGURE OUT WHAT KNODE DOES
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_nn_weibull
from wfpt import wiener_like_nn_angle 
from wfpt import wiener_like_nn_ddm
from wfpt import wiener_like_nn_levy
from wfpt import wiener_like_nn_ornstein

class HDDMnn_new(HDDM):
    """HDDM model that uses WEIBULL neural net likelihood

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        #self.free = kwargs.pop('free', False) # 
        self.model = kwargs.pop('model', 'weibull')

        if self.model == 'ddm':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ddm', wienernn_like_ddm)
        
        if self.model == 'weibull' or self.model == 'weibull_cdf':
            self.wfpt_nn = stochastic_from_dist('Wienernn_weibull', wienernn_like_weibull)
        
        if self.model == 'angle':
            self.wfpt_nn = stochastic_from_dist('Wienernn_angle', wienernn_like_angle) 

        if self.model == 'levy':
            self.wfpt_nn = stochastic_from_dist('Wienernn_levy', wienernn_like_levy) 

        if self.model == 'ornstein':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ornstein', wienernn_like_ornstein)


        #self.wfpt_nn_new_class = Wienernn_new # attach corresponding likelihood
        #self.k = kwargs.pop('k', False)
        super(HDDMnn_new, self).__init__(*args, **kwargs)
    
    def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()
        # SPLIT BY MODEL TO ACCOMMODATE TRAINED PARAMETER BOUNDS BY MODEL
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
                                                               std_upper = 1
                                                               ))
            if 'beta' in include:
                knodes.update(self._create_family_trunc_normal('beta', 
                                                               lower = 0.31, 
                                                               upper = 6.99, 
                                                               value = 3.34,
                                                               std_upper = 1
                                                               ))

        if self.model == 'ddm':
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

        wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0 #self.default_intervars['sv']
        wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0 #self.default_intervars['sz']
        wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0 #self.default_intervars['st']
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else 0 #self.p_outlier
        
        # MODEL SPECIFIC PARAMETERS
        if self.model == 'weibull' or self.model == 'weibull_cdf':
            wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 3 
            wfpt_parents['beta'] = knodes['beta_bottom'] if 'beta' in self.include else 3
        if self.model == 'ornstein':
            wfpt_parents['g'] = knodes['g_bottom'] if 'g' in self.include else 0
        if self.model == 'levy':
            wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 2
        if self.model == 'angle':
            wfpt_parents['theta'] = knodes['theta_bottom'] if 'theta' in self.include else 0

        print('wfpt parents: ')
        print(wfpt_parents)
        return wfpt_parents


    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        
        return Knode(self.wfpt_nn, 
                     'wfpt', 
                     observed = True, 
                     col_name = ['nn_response', 'rt'], # TODO: One could preprocess at initialization
                     **wfpt_parents)

def wienernn_like_weibull(x, 
                          v, 
                          sv, 
                          a, 
                          alpha,
                          beta,
                          z, 
                          sz, 
                          t, 
                          st, 
                          p_outlier = 0): #theta

    wiener_params = {'err': 1e-4, # 
                     'n_st': 2, #
                     'n_sz': 2, # 
                     'use_adaptive': 1, #
                     'simps_err': 1e-3, # 
                     'w_outlier': 0.1}
    #wp = wiener_params

    return wiener_like_nn_weibull(np.absolute(x['rt'].values).astype(np.float32),
                                  x['nn_response'].values.astype(np.float32), 
                                  v, 
                                  sv,
                                  a, 
                                  alpha, 
                                  beta,
                                  z, 
                                  sz,
                                  t, 
                                  st, 
                                  p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                  **wiener_params)

def wienernn_like_levy(x, 
                       v, 
                       sv, 
                       a, 
                       alpha,
                       z, 
                       sz, 
                       t, 
                       st, 
                       p_outlier = 0): #theta

    wiener_params = {'err': 1e-4, # 
                     'n_st': 2, #
                     'n_sz': 2, # 
                     'use_adaptive': 1, #
                     'simps_err': 1e-3, # 
                     'w_outlier': 0.1}
    
   #wp = wiener_params

    return wiener_like_nn_levy(np.absolute(x['rt'].values).astype(np.float32),
                               x['nn_response'].values.astype(np.float32), 
                               v, 
                               sv,
                               a, 
                               alpha, 
                               z, 
                               sz,
                               t, 
                               st, 
                               p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                               **wiener_params)

def wienernn_like_ornstein(x, 
                           v, 
                           sv, 
                           a, 
                           g,
                           z, 
                           sz, 
                           t, 
                           st, 
                           p_outlier = 0): #theta
    
    wiener_params = {'err': 1e-4, # 
                     'n_st': 2, #
                     'n_sz': 2, # 
                     'use_adaptive': 1, #
                     'simps_err': 1e-3, # 
                     'w_outlier': 0.1}
    
    #wp = wiener_params

    return wiener_like_nn_ornstein(np.absolute(x['rt'].values).astype(np.float32),
                                   x['nn_response'].values.astype(np.float32), 
                                   v, 
                                   sv,
                                   a, 
                                   g, 
                                   z, 
                                   sz,
                                   t, 
                                   st, 
                                   p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                   **wiener_params)

def wienernn_like_ddm(x, 
                      v, 
                      sv, 
                      a, 
                      z, 
                      sz, 
                      t, 
                      st, 
                      p_outlier = 0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    #wp = wiener_params

    #nn_response = x['nn_response'].values.astype(int)
    
    return wiener_like_nn_ddm(np.absolute(x['rt'].values).astype(np.float32),
                              x['nn_response'].values.astype(np.float32),  
                              v, 
                              sv, 
                              a, 
                              z, 
                              sz, 
                              t, 
                              st, 
                              p_outlier = p_outlier,
                              **wiener_params)

def wienernn_like_angle(x, 
                        v, 
                        sv, 
                        a, 
                        theta, 
                        z, 
                        sz,
                        t, 
                        st, 
                        p_outlier = 0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    #wp = wiener_params

    #nn_response = x['nn_response'].values.astype(int)
    
    return wiener_like_nn_angle(np.absolute(x['rt'].values).astype(np.float32),
                                x['nn_response'].values.astype(np.float32),  
                                v, 
                                sv, 
                                a, 
                                theta,
                                z, 
                                sz, 
                                t, 
                                st, 
                                p_outlier = p_outlier,
                                **wiener_params)
#Wienernn_angle = 

# TODO CHECK WHAT THIS IS EVEN DOING

#Wienernn_new = stochastic_from_dist('Wienernn_new', wienernn_like_new)



            
    # def _create_stochastic_knodes(self, include):
    #     knodes = super(HDDMnn_weibull, self)._create_stochastic_knodes(include) # 
    #     if self.free:
    #         knodes.update(self._create_family_gamma_gamma_hnormal('beta', 
    #                                                               g_mean = 1.5, 
    #                                                               g_std = 0.75,
    #                                                               std_std = 2, 
    #                                                               std_value = 0.1,
    #                                                               value = 1)) # TODO: Check if this is a good prior
    #         if self.k:
    #             knodes.update(self._create_family_gamma_gamma_hnormal('alpha', 
    #                                                                   g_mean = 1.5, 
    #                                                                   g_std = 0.75, 
    #                                                                   std_std = 2,
    #                                                                   std_value = 0.1,
    #                                                                   value = 1)) # TODO: Check if this is a good prior
    #     else:
    #         knodes.update(self._create_family_trunc_normal('beta', 
    #                                                        lower = 0.31, 
    #                                                        upper = 6.99, 
    #                                                        value = 3.34,
    #                                                        std_upper = 1))
    #         if self.k:
    #             knodes.update(self._create_family_trunc_normal('alpha',
    #                                                            lower = 0.31, 
    #                                                            upper = 4.99, 
    #                                                            value = 2.34,
    #                                                            std_upper = 1))
    #     return knodes

    # TODO: CLARIFY WHAT THIS FUNCTION DOES

        # def _create_wfpt_parents_dict(self, knodes):
    #     print(knodes)
    #     wfpt_parents = super(HDDMnn_new, self)._create_wfpt_parents_dict(knodes)
    #     print(wfpt_parents)
    #     wfpt_parents['beta'] = knodes['beta_bottom']
    #     wfpt_parents['alpha'] = knodes['alpha_bottom'] if self.k else 3.00
    #     return wfpt_parents

    