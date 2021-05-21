
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rl
from collections import OrderedDict
import hddm

class Hrl(HDDM):
    """RL model that can be used to analyze data from two-armed bandit tasks.

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        # self.dual = kwargs.pop('dual', False)
        # self.alpha = kwargs.pop('alpha', True)
        # self.z = kwargs.pop('z', False)
        self.rl_class = RL_like_MAB

        super(Hrl, self).__init__(*args, **kwargs)
    
    def plot_posteriors(self, **kwargs):
        hddm.utils.plot_posteriors_az(self)

    def _create_stochastic_knodes(self, include):
        params = ['beta']
        # if 'p_outlier' in self.include:
        #     params.append('p_outlier')
        # if 'z' in self.include:
        #     params.append('z')
        include = set(params)

        knodes = super(Hrl, self)._create_stochastic_knodes(include)
        if self.non_centered:
            print('setting learning rate parameter(s) to be non-centered')
            if self.alpha:
                knodes.update(self._create_family_normal_non_centered(
                    'alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.dual:
                knodes.update(self._create_family_normal_non_centered(
                    'pos_alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        else:
            # if self.alpha:
            # knodes.update(self._create_family_normal(
            #    'alpha', value=0.5, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            knodes.update(self._create_family_normal(
                'beta', value=0.5, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            # if self.dual:
            #     knodes.update(self._create_family_normal(
            #         'pos_alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        #wfpt_parents['v'] = knodes['v_bottom']
        #wfpt_parents['alpha'] = knodes['alpha_bottom']
        wfpt_parents['beta'] = knodes['beta_bottom']
        #wfpt_parents['pos_alpha'] = knodes['pos_alpha_bottom'] if self.dual else 100.00
        #wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.rl_class, 'wfpt', observed=True, col_name=['split_by', 'feedback', 'response', 'q_init'], **wfpt_parents)


def RL_like(x, v, alpha, pos_alpha, z=0.5, p_outlier=0):
    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    sum_logp = 0
    wp = wiener_params
    response = x['response'].values.astype(int)
    q = x['q_init'].iloc[0]
    feedback = x['feedback'].values.astype(float)
    split_by = x['split_by'].values.astype(int)
    return wiener_like_rl(response, feedback, split_by, q, alpha, pos_alpha, v, z, p_outlier=p_outlier, **wp)
RL = stochastic_from_dist('RL', RL_like)


# Multi(two)-armed bandit
def RL_like_MAB(x, beta):
    response = x['response'].values.astype(int)
    q = x['q_init'].iloc[0]
    feedback = x['feedback'].values.astype(float)
    split_by = x['split_by'].values.astype(int)

    k = 2 # number of bandits/actions
    s_size = response.shape[0]
    qs = np.zeros(k) # q-values
    num_act = np.zeros(k) # keep count of every action
    np_eps = 2.718281828459

    sum_logp = 0
    
    # loop through all trials in current condition
    for i in range(0, s_size):
        a = response[i]

        # Compute softmax 
        action_values = qs[:]
        exp_act_val = np.power([np_eps]*len(action_values), beta * action_values)
        deno = np.sum(exp_act_val)
        prob_tuple = exp_act_val/deno

        # q-value update
        num_act[a] += 1
        qs[a] = qs[a] + (feedback[i]-qs[a])/num_act[a]

        # If one probability = 0, the log sum will be -Inf
        flag_inf = False
        if 0 in prob_tuple:
            flag_inf = True
        if beta <= 0:
            flag_inf = True

        sum_logp += np.log(prob_tuple[a])

    if flag_inf == True:
        return -np.inf

    return sum_logp

RL_like_MAB = stochastic_from_dist('RL_MAB', RL_like_MAB)