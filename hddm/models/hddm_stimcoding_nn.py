from copy import copy
import numpy as np
from collections import OrderedDict

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM

from wfpt import wiener_like_nn_weibull
from wfpt import wiener_like_nn_angle 
from wfpt import wiener_like_nn_ddm
from wfpt import wiener_like_nn_ddm_analytic
from wfpt import wiener_like_nn_levy
from wfpt import wiener_like_nn_ornstein
from wfpt import wiener_like_nn_ddm_sdv
from wfpt import wiener_like_nn_ddm_sdv_analytic
from wfpt import wiener_like_nn_full_ddm

class HDDMnnStimCoding(HDDM):
    """HDDM model that can be used when stimulus coding and estimation
    of bias (i.e. displacement of starting point z) is required.

    In that case, the 'resp' column in your data should contain 0 and
    1 for the chosen stimulus (or direction), not whether the response
    was correct or not as you would use in accuracy coding. You then
    have to provide another column (referred to as stim_col) which
    contains information about which the correct response was.

    :Arguments:
        split_param : {'v', 'z'} <default='z'>
            There are two ways to model stimulus coding in the case where both stimuli
            have equal information (so that there can be no difference in drift):
            * 'z': Use z for stimulus A and 1-z for stimulus B
            * 'v': Use drift v for stimulus A and -v for stimulus B

        stim_col : str
            Column name for extracting the stimuli to use for splitting.

        drift_criterion : bool <default=False>
            Whether to estimate a constant factor added to the drift-rate.
            Requires split_param='v' to be set.

    """
    def __init__(self, *args, **kwargs):
        self.nn = True
        self.stim_col = kwargs.pop('stim_col', 'stim')
        self.split_param = kwargs.pop('split_param', 'z')
        self.drift_criterion = kwargs.pop('drift_criterion', False)
        self.model = kwargs.pop('model', 'weibull')
        self.model = kwargs.pop('w_outlier', 0.1)

        print(kwargs['include'])
        # Attach likelihood corresponding to model
        if self.model == 'ddm':
            self.wfpt_nn = stochastic_from_dist('Wienernn_ddm', wienernn_like_ddm)

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

        if self.split_param == 'z':
            assert not self.drift_criterion, "Setting drift_criterion requires split_param='v'."
            print("Setting model to be non-informative")
            kwargs['informative'] = False

            # Add z if it is split parameter but not included in 'include'
            if 'include' in kwargs and 'z' not in kwargs['include']:
                kwargs['include'].append('z')
            else:
                print('passing through here...')
                kwargs['include'] = ['z']
            print("Adding z to includes.")

        self.stims = np.asarray(np.sort(np.unique(args[0][self.stim_col])))
        assert len(self.stims) == 2, "%s must contain two stimulus types" % self.stim_col

        super(HDDMnnStimCoding, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):

        #def _create_stochastic_knodes(self, include):
        knodes = OrderedDict()

        # PARAMETERS COMMON TO ALL MODELS
        if 'p_outlier' in include:
            knodes.update(self._create_family_invlogit('p_outlier',
                                                        value = 0.2,
                                                        g_tau = 10**-2,
                                                        std_std = 0.5
                                                        ))

        if self.drift_criterion:
            knodes.update(self._create_family_normal_normal_hnormal('dc',
                                                                     value = 0,
                                                                     g_mu = 0,
                                                                     g_tau = 3**-2,
                                                                     std_std = 2))
        
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

        return knodes

    # def _create_stochastic_knodes(self, include):
    #     knodes = super(HDDMStimCoding, self)._create_stochastic_knodes(include)
        
    #     if self.drift_criterion:
    #         # Add drift-criterion parameter
    #         knodes.update(self._create_family_normal_normal_hnormal('dc',
    #                                                                 value = 0,
    #                                                                 g_mu = 0,
    #                                                                 g_tau = 3**-2,
    #                                                                 std_std = 2))

    #     return knodes

    def _create_wfpt_parents_dict(self, knodes):
        print(knodes)
        wfpt_parents = OrderedDict()
        wfpt_parents['a'] = knodes['a_bottom']
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['t'] = knodes['t_bottom']
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        wfpt_parents['p_outlier'] = knodes['p_outlier_bottom'] if 'p_outlier' in self.include else 0 #self.p_outlier
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


        # SPECIFIC TO STIMCODING
        if self.drift_criterion: 
            wfpt_parents['dc'] = knodes['dc_bottom']

        print('wfpt parents: ')
        print(wfpt_parents)
        return wfpt_parents

    # def _create_wfpt_parents_dict(self, knodes):
    #     wfpt_parents = super(HDDMStimCoding, self)._create_wfpt_parents_dict(knodes)
        
    #     if self.drift_criterion:
    #         wfpt_parents['dc'] = knodes['dc_bottom']
        
    #     return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        
        # Here we use a special Knode (see below) that either inverts v or z
        # depending on what the correct stimulus was for that trial type.
        
        return KnodeWfptStimCoding(self.wfpt_nn, 
                                   'wfpt', # TD: ADD wfpt class we need
                                   observed = True, 
                                   col_name = ['response', 'rt'], # col_name = 'rt',
                                   depends = [self.stim_col],
                                   split_param = self.split_param,
                                   stims = self.stims,
                                   stim_col = self.stim_col,
                                   **wfpt_parents)

class KnodeWfptStimCoding(Knode):
    def __init__(self, *args, **kwargs):
        self.split_param = kwargs.pop('split_param')
        self.stims = kwargs.pop('stims')
        self.stim_col = kwargs.pop('stim_col')

        super(KnodeWfptStimCoding, self).__init__(*args, **kwargs)

    def create_node(self, name, kwargs, data):
        
        # the addition of "depends=['stim']" in the call of
        # KnodeWfptInvZ in HDDMStimCoding makes that data are
        # submitted splitted by the values of the variable stim the
        # following lines check if the variable stim is equal to the
        # value of stim for which z' = 1-z and transforms z if this is
        # the case (similar to v)

        dc = kwargs.pop('dc', None)
        
        if all(data[self.stim_col] == self.stims[1]): # AF NOTE: Reversed this, previously self.stims[0], compare what is expected as data to my simulator...
            if self.split_param == 'z':
                kwargs['z'] = 1 - kwargs['z']
            elif self.split_param == 'v' and dc is None:
                kwargs['v'] = - kwargs['v']
            elif self.split_param == 'v' and dc != 0:
                kwargs['v'] = - kwargs['v'] + dc # 
            else:
                raise ValueError('split_var must be either v or z, but is %s' % self.split_var)

            return self.pymc_node(name, **kwargs)
        else:
            if dc is not None:
                kwargs['v'] = kwargs['v'] + dc
            return self.pymc_node(name, **kwargs)

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