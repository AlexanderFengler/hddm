
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
from wfpt import wiener_like_nn_angle #TODO
from wfpt import wiener_like_nn_ddm

class HDDMnn_new(HDDM):
    """HDDM model that uses WEIBULL neural net likelihood

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.free = kwargs.pop('free', False) # 
        self.k = kwargs.pop('k', False)
        self.model = kwargs.pop('model', 'weibull')
        #self.wfpt_nn_new_class = Wienernn_new # attach corresponding likelihood
        self.wfpt_nn_weibull_class = stochastic_from_dist('Wienernn_weibull', wienernn_like_weibull)

        super(HDDMnn_new, self).__init__(*args, **kwargs)
    
    # def _create_stochastic_knodes(self, include):
    #     knodes = OrderedDict()
    #     print(self.model)

    #     if 'a' in include:

    def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()

        if 'a' in include:
            knodes.update(self._create_family_trunc_normal('a',
                                                           lower = 0.3,
                                                           upper = 2,
                                                           value = 1,
                                                           std_upper = 1 # added AF
                                                           ))
        if 'v' in include:
            knodes.update(self._create_family_trunc_normal('v', 
                                                           lower = - 2.7,
                                                           upper = 2.7,
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
                                                           lower = 0.31, 
                                                           upper = 4.99, 
                                                           value = 2.34,
                                                           std_upper = 1
                                                           ))
        if 'beta' in include:
            nodes.update(self._create_family_trunc_normal('beta', 
                                                           lower = 0.31, 
                                                           upper = 6.99, 
                                                           value = 3.34,
                                                           std_upper = 1
                                                           ))
        return knodes


            
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
    
    def _create_wfpt_parents_dict(self, knodes):
        print(knodes)
        wfpt_parents = super(HDDMnn_new, self)._create_wfpt_parents_dict(knodes)
        print(wfpt_parents)
        wfpt_parents['beta'] = knodes['beta_bottom']
        wfpt_parents['alpha'] = knodes['alpha_bottom'] if self.k else 3.00
        return wfpt_parents

    # TODO: CLARIFY WHAT THIS FUNCTION DOES
    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_nn_weibull_class, 
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

# TODO CHECK WHAT THIS IS EVEN DOING

#Wienernn_new = stochastic_from_dist('Wienernn_new', wienernn_like_new)