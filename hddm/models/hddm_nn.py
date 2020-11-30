
"""
"""
import hddm
from collections import OrderedDict
from copy import copy
import numpy as np
import pymc
import wfpt
#import pickle
from functools import partial

from kabuki.hierarchical import Knode # LOOK INTO KABUKI TO FIGURE OUT WHAT KNODE EXACTLY DOES
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from hddm.keras_models import load_mlp
from hddm.cnn.wrapper import load_cnn


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
        self.model = kwargs.pop('model', 'weibull')
        self.nbin = kwargs.pop('nbin', 512)
        
        # Load Network and likelihood function
        if self.network_type == 'mlp':
                self.network = load_mlp(model = self.model)
                network_dict = {'network': self.network}
                likelihood_ = hddm.likelihoods_mlp.make_mlp_likelihood(model = self.model)

        if self.network_type == 'cnn':
                self.network = load_cnn(model = self.model, nbin=self.nbin)
                network_dict = {'network': self.network}
                likelihood_ = hddm.likelihoods_cnn.make_cnn_likelihood(model = self.model)
                #partial(wrapper, specific_forward_pass)

        # Make model specific likelihood
        self.wfpt_nn = stochastic_from_dist('Wienernn' + '_' + self.model,
                                            partial(likelihood_, **network_dict))
        # Initialize super class
        super(HDDMnn, self).__init__(*args, **kwargs)
        print(self.p_outlier)
    
    # def _create_stochastic_knodes(self, include):
    #     knodes = OrderedDict()
        
    #     # SPLIT BY MODEL TO ACCOMMODATE TRAINED PARAMETER BOUNDS BY MODEL

    #     # PARAMETERS COMMON TO ALL MODELS
    #     if 'p_outlier' in include:
    #         knodes.update(self._create_family_invlogit('p_outlier',
    #                                                     value = 0.2,
    #                                                     g_tau = 10**-2,
    #                                                     std_std = 0.5
    #                                                     ))

    #     if self.model == 'weibull' or self.model == 'weibull_cdf':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.5,
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 2.5,
    #                                                            upper = 2.5,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        )) # should have lower = 0.2, upper = 0.8
    #         if 'alpha' in include:
    #             knodes.update(self._create_family_trunc_normal('alpha',
    #                                                            lower = 0.31, 
    #                                                            upper = 4.99, 
    #                                                            value = 2.34,
    #                                                            std_upper = 2
    #                                                            ))
    #         if 'beta' in include:
    #             knodes.update(self._create_family_trunc_normal('beta', 
    #                                                            lower = 0.31, 
    #                                                            upper = 6.99, 
    #                                                            value = 3.34,
    #                                                            std_upper = 2
    #                                                            ))

    #     if self.model == 'weibull_cdf_concave':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.5,
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 2.5,
    #                                                            upper = 2.5,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        )) # should have lower = 0.2, upper = 0.8
    #         if 'alpha' in include:
    #             knodes.update(self._create_family_trunc_normal('alpha',
    #                                                            lower = 1.00, # this guarantees initial concavity of the likelihood
    #                                                            upper = 4.99, 
    #                                                            value = 2.34,
    #                                                            std_upper = 2
    #                                                            ))
    #         if 'beta' in include:
    #             knodes.update(self._create_family_trunc_normal('beta', 
    #                                                            lower = 0.31, 
    #                                                            upper = 6.99, 
    #                                                            value = 3.34,
    #                                                            std_upper = 2
    #                                                            ))

    #     if self.model == 'ddm' or self.model == 'ddm_analytic':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.5,
    #                                                            value = 1.4,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 3.0,
    #                                                            upper = 3.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        )) # should have lower = 0.1, upper = 0.9

    #         print(knodes.keys())

        
    #     if self.model == 'ddm_sdv' or self.model == 'ddm_sdv_analytic':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.5,
    #                                                            value = 1.4,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 3.0,
    #                                                            upper = 3.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        )) # should have lower = 0.1, upper = 0.9
    #         if 'sv' in include:
    #             knodes.update(self._create_family_trunc_normal('sv', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2.5, 
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))

    #     if self.model == 'full_ddm' or self.model == 'full_ddm2':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.5,
    #                                                            value = 1.4,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 3.0,
    #                                                            upper = 3.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 0.25,
    #                                                            upper = 2.25, 
    #                                                            value = .5,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        )) # should have lower = 0.1, upper = 0.9

    #         if 'sz' in include:
    #             knodes.update(self._create_family_trunc_normal('sz', 
    #                                                            lower = 1e-3,
    #                                                            upper = 0.2, 
    #                                                            value = 0.1,
    #                                                            std_upper = 0.1 # added AF
    #                                                            ))

    #         if 'sv' in include:
    #             knodes.update(self._create_family_trunc_normal('sv', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2.0, 
    #                                                            value = 1.0,
    #                                                            std_upper = 0.5 # added AF
    #                                                            ))

    #         if 'st' in include:
    #             knodes.update(self._create_family_trunc_normal('st', 
    #                                                            lower = 1e-3,
    #                                                            upper = 0.25, 
    #                                                            value = 0.125,
    #                                                            std_upper = 0.1 # added AF
    #                                                            ))

    #     if self.model == 'angle':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.0,
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 3.0,
    #                                                            upper = 3.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        ))
    #         if 'theta' in include:
    #             knodes.update(self._create_family_trunc_normal('theta',
    #                                                            lower = -0.1, 
    #                                                            upper = 1.45, 
    #                                                            value = 0.5,
    #                                                            std_upper = 1
    #                                                            )) # should have lower = 0.2, upper = 0.8

    #     if self.model == 'ornstein':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.0,
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 2.0,
    #                                                            upper = 2.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        ))
    #         if 'g' in include:
    #             knodes.update(self._create_family_trunc_normal('g',
    #                                                            lower = -1.0, 
    #                                                            upper = 1.0, 
    #                                                            value = 0.5,
    #                                                            std_upper = 1
    #                                                            )) # should have lower = 0.2, upper = 0.8
        
    #     if self.model == 'levy':
    #         if 'a' in include:
    #             knodes.update(self._create_family_trunc_normal('a',
    #                                                            lower = 0.3,
    #                                                            upper = 2.0,
    #                                                            value = 1,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'v' in include:
    #             knodes.update(self._create_family_trunc_normal('v', 
    #                                                            lower = - 3.0,
    #                                                            upper = 3.0,
    #                                                            value = 0,
    #                                                            std_upper = 1.5
    #                                                            ))
    #         if 't' in include:
    #             knodes.update(self._create_family_trunc_normal('t', 
    #                                                            lower = 1e-3,
    #                                                            upper = 2, 
    #                                                            value = .01,
    #                                                            std_upper = 1 # added AF
    #                                                            ))
    #         if 'z' in include:
    #             knodes.update(self._create_family_invlogit('z',
    #                                                        value = .5,
    #                                                        g_tau = 10**-2,
    #                                                        std_std = 0.5
    #                                                        ))
    #         if 'alpha' in include:
    #             knodes.update(self._create_family_trunc_normal('alpha',
    #                                                            lower = 1.0, 
    #                                                            upper = 2.0, 
    #                                                            value = 1.5,
    #                                                            std_upper = 1
    #                                                            ))
    #                                                            # should have lower = 0.1, upper = 0.9
                      
    #     print('knodes')
    #     print(knodes)

    #     return knodes

    # def _create_wfpt_parents_dict(self, knodes):
    #     wfpt_parents = OrderedDict()
    #     wfpt_parents['a'] = knodes['a_bottom']
    #     wfpt_parents['v'] = knodes['v_bottom']
    #     wfpt_parents['t'] = knodes['t_bottom']
    #     wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5
    #     wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else self.p_outlier
    #     wfpt_parents['w_outlier'] = self.w_outlier # likelihood of an outlier point

    #     # MODEL SPECIFIC PARAMETERS
    #     if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull_cdf_concave':
    #         wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 3 
    #         wfpt_parents['beta'] = knodes['beta_bottom'] if 'beta' in self.include else 3
        
    #     if self.model == 'ornstein':
    #         wfpt_parents['g'] = knodes['g_bottom'] if 'g' in self.include else 0
        
    #     if self.model == 'levy':
    #         wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 2
        
    #     if self.model == 'angle':
    #         wfpt_parents['theta'] = knodes['theta_bottom'] if 'theta' in self.include else 0

    #     if self.model == 'full_ddm' or self.model =='full_ddm2':
    #         wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0 #self.default_intervars['sv']
    #         wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0 #self.default_intervars['sz']
    #         wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0 #self.default_intervars['st']

    #     print('wfpt parents: ')
    #     print(wfpt_parents)
    #     return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_nn, 
                     'wfpt', 
                     observed = True, 
                     col_name = ['response', 'rt'], # TODO: One could preprocess at initialization
                     **wfpt_parents)

# UNUSED -------------------------------------------------------------
        
# if self.model == 'ddm':
#     # Previous
#     #self.wfpt_nn = generate_wfpt_stochastic_class()
#     if self.network_type == 'mlp':
#         pass
#         #self.wfpt_nn = stochastic_from_dist('Wienernn_ddm', partial(wienernn_like_ddm, **network_dict))
#     #likelihood_fun = wienernn_like_ddm
#     print('Loaded MLP Likelihood for ', self.model, ' model!')
        
# if self.model == 'ddm_sdv':
#     if self.network_type == 'mlp':
#         self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_sdv', partial(wienernn_like_ddm_sdv, **network_dict))

# if self.model == 'ddm_analytic':
#     if self.network_type == 'mlp':
#         self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_analytic', partial(wienernn_like_ddm_analytic, **network_dict))

# if self.model == 'ddm_sdv_analytic':
#     if self.network_type == 'mlp':
#         self.wfpt_nn = stochastic_from_dist('Wienernn_ddm_sdv_analytic', partial(wienernn_like_ddm_sdv_analytic, **network_dict))

# if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull_cdf_concave':
#     if self.network_type == 'mlp':
#         self.wfpt_nn = stochastic_from_dist('Wienernn_weibull', wienernn_like_weibull)

# if self.model == 'angle':
#     self.wfpt_nn = stochastic_from_dist('Wienernn_angle', wienernn_like_angle) 

# if self.model == 'levy':
#     self.wfpt_nn = stochastic_from_dist('Wienernn_levy', wienernn_like_levy) 

# if self.model == 'ornstein':
#     self.wfpt_nn = stochastic_from_dist('Wienernn_ornstein', wienernn_like_ornstein)

# if self.model == 'full_ddm' or self.model == 'full_ddm2':
#     self.wfpt_nn = stochastic_from_dist('Wienernn_full_ddm', wienernn_like_full_ddm)

# ------------------------------------------------------------------------