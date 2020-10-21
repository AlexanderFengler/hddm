
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt
import pickle

from kabuki.hierarchical import Knode # LOOK INTO KABUKI TO FIGURE OUT WHAT KNODE DOES
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_nn_new #TODO

class HDDMnn_new(HDDM):
    """HDDM model that uses NEW neural net likelihood

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.free = kwargs.pop('free', False) # 
        self.k = kwargs.pop('k', False)
        self.wfpt_nn_new_class = Wienernn_new # attach corresponding likelihood

        super(HDDMnn_new, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMnn_new, self)._create_stochastic_knodes(include) # 
        if self.free:
            knodes.update(self._create_family_gamma_gamma_hnormal('beta', 
                                                                  g_mean = 1.5, 
                                                                  g_std = 0.75,
                                                                  std_std = 2, 
                                                                  std_value = 0.1,
                                                                  value = 1)) # TODO: Check if this is a good prior
            if self.k:
                knodes.update(self._create_family_gamma_gamma_hnormal('alpha', 
                                                                      g_mean = 1.5, 
                                                                      g_std = 0.75, 
                                                                      std_std = 2,
                                                                      std_value = 0.1,
                                                                      value = 1)) # TODO: Check if this is a good prior
        else:
            knodes.update(self._create_family_trunc_normal('beta', 
                                                           lower = 0.3, 
                                                           upper = 7, 
                                                           value = 1,
                                                           std_upper = 1))
            if self.k:
                knodes.update(self._create_family_trunc_normal('alpha',
                                                               lower = 0.3, 
                                                               upper = 5, 
                                                               value = 1,
                                                               std_upper = 1))
        return knodes

    # TODO: CLARIFY WHAT THIS FUNCTION DOES
    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnn_new, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['beta'] = knodes['beta_bottom']
        wfpt_parents['alpha'] = knodes['alpha_bottom'] if self.k else 3.00
        return wfpt_parents

    # TODO: CLARIFY WHAT THIS FUNCTION DOES
    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_nn_new_class, 
                    'wfpt', 
                     observed = True, 
                     col_name = ['nn_response', 'rt'], # TODO: One could preprocess at initialization
                     **wfpt_parents)

def wienernn_like_new(x, 
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
    wp = wiener_params

    #with open("weights.pickle", "rb") as tmp_file:
    #    weights = pickle.load(tmp_file)
    #with open('biases.pickle', 'rb') as tmp_file:
    #    biases = pickle.load(tmp_file)
    #with open('activations.pickle', 'rb') as tmp_file:
    #    activations = pickle.load(tmp_file)

    #print('hei')
    #nn_response =  # TODO: just request this to be integer to begin with ?!
    
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
                                  **wp)

# TODO CHECK WHAT THIS IS EVEN DOING

Wienernn_new = stochastic_from_dist('Wienernn_new', wienernn_like_new)