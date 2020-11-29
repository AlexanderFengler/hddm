from collections import OrderedDict
from copy import deepcopy
import math
import numpy as np
import pymc as pm
import pandas as pd
from patsy import dmatrix
#import pickle

import hddm
from hddm.models import HDDM
from hddm.keras_models import load_mlp
import kabuki
from kabuki import Knode
from kabuki.utils import stochastic_from_dist
import kabuki.step_methods as steps
from functools import partial
import wfpt

##########################################################
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

    return stoch

################################################################################################
# Defining only the model likelihood at this point !
def generate_wfpt_nn_full_ddm_reg_stochastic_class(wiener_params = None,
                                                   sampling_method = 'cdf',
                                                   cdf_range = (-5, 5), 
                                                   sampling_dt = 1e-4):

    def wiener_multi_like_nn_full_ddm(value, v, sv, a, z, sz, t, st, 
                                      reg_outcomes, 
                                      p_outlier = 0, 
                                      w_outlier = 0.1):

        """Log-likelihood for the full DDM using the interpolation method"""

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
                                                       w_outlier = w_outlier)

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

    stoch = stochastic_from_dist('wfpt_reg', wiener_multi_like_nn_full_ddm)
    stoch.random = random

    return stoch

# Defining only the model likelihood at this point !
def generate_wfpt_nn_angle_reg_stochastic_class(wiener_params = None,
                                                sampling_method = 'cdf',
                                                cdf_range = (-5, 5), 
                                                sampling_dt = 1e-4):

    def wiener_multi_like_nn_angle(value, v, a, theta, z, t, 
                                   reg_outcomes, 
                                   p_outlier = 0, 
                                   w_outlier = 0.1):

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
                                                    w_outlier = w_outlier)

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

    stoch = stochastic_from_dist('wfpt_reg', wiener_multi_like_nn_angle)
    stoch.random = random

    return stoch

# Defining only the model likelihood at this point !
def generate_wfpt_nn_levy_reg_stochastic_class(wiener_params = None,
                                               sampling_method = 'cdf',
                                               cdf_range = (-5, 5), 
                                               sampling_dt = 1e-4):

    #set wiener_params
    if wiener_params is None:
        wiener_params = {'err': 1e-4,
                         'n_st': 2, 
                         'n_sz': 2,
                         'use_adaptive': 1,
                         'simps_err': 1e-3,
                         'w_outlier': 0.1}
    
    wp = wiener_params

    def wiener_multi_like_nn_levy(value, v, a, alpha, z, t, 
                                  reg_outcomes, 
                                  p_outlier = 0, 
                                  w_outlier = 0.1):

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
                                                   w_outlier = w_outlier)

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

    stoch = stochastic_from_dist('wfpt_reg', wiener_multi_like_nn_levy)
    stoch.random = random

    return stoch

#wfpt_reg_like = generate_wfpt_nn_reg_stochastic_class(sampling_method = 'drift')
################################################################################################


# Defining only the model likelihood at this point !
def generate_wfpt_nn_ornstein_reg_stochastic_class(wiener_params = None,
                                                sampling_method = 'cdf',
                                                cdf_range = (-5, 5), 
                                                sampling_dt = 1e-4):

    def wiener_multi_like_nn_ornstein(value, v, a, g, z, t, 
                                      reg_outcomes, 
                                      p_outlier = 0, 
                                      w_outlier = 0.1):

        """Log-likelihood for the full DDM using the interpolation method"""

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
                                                       w_outlier = w_outlier)


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

    stoch = stochastic_from_dist('wfpt_reg', wiener_multi_like_nn_ornstein)
    stoch.random = random

    return stoch
################################################################################################

# Defining only the model likelihood at this point !
def generate_wfpt_nn_weibull_reg_stochastic_class(wiener_params = None,
                                                  sampling_method = 'cdf',
                                                  cdf_range = (-5, 5), 
                                                  sampling_dt = 1e-4):

    def wiener_multi_like_nn_weibull(value, v, a, alpha, beta, z, t, 
                                     reg_outcomes, 
                                     p_outlier = 0, 
                                     w_outlier = 0.1):

        """Log-likelihood for the full DDM using the interpolation method"""

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
                                                      w_outlier = w_outlier)


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

    stoch = stochastic_from_dist('wfpt_reg', wiener_multi_like_nn_weibull)
    stoch.random = random

    return stoch
################################################################################################

class KnodeRegress(kabuki.hierarchical.Knode):
    def __init__(self, *args, **kwargs):
        
        # Whether or not to keep the regressor trace
        self.keep_regressor_trace = kwargs.pop('keep_regressor_trace', False)
        
        # Initialize kabuke.hierarchical.Knode
        super(KnodeRegress, self).__init__(*args, **kwargs)

    def create_node(self, name, kwargs, data):
        
        reg = kwargs['regressor']
        
        # order parents according to user-supplied args
        # QAF: what's the point of that?

        args = []
        for arg in reg['params']:
            for parent_name, parent in kwargs['parents'].items():
                if parent_name == arg:
                    args.append(parent)

        parents = {'args': args}

        # Make sure design matrix is kosher
        dm = dmatrix(reg['model'], data = data)
        if math.isnan(dm.sum()):
            raise NotImplementedError('DesignMatrix contains NaNs.')

        def func(args, 
                 design_matrix = dmatrix(reg['model'], data = data), 
                 link_func = reg['link_func']):
            
            # convert parents to matrix
            params = np.matrix(args)
            
            # Apply design matrix to input data
            if design_matrix.shape[1] != params.shape[1]:
                raise NotImplementedError('Missing columns in design matrix. You need data for all conditions for all subjects.')
            
            # Get 'predictors' ( wfpt model parameters according to current regression parameterization)
            predictor = link_func(pd.DataFrame((design_matrix * params).sum(axis = 1), # compute regression matrix algebra
                                                index = data.index)) # QAF why data.index, is this necessary ?

            return pd.DataFrame(predictor, index = data.index)

        return self.pymc_node(func, 
                              kwargs['doc'], 
                              name, 
                              parents = parents,
                              trace = self.keep_regressor_trace)

class HDDMnnRegressor(HDDM):
    """HDDMnnRegressor allows estimation of the NNDDM where parameter
    values are linear models of a covariate (e.g. a brain measure like
    fMRI or different conditions).
    """

    def __init__(self, data, models, model = 'ddm', group_only_regressors = True, keep_regressor_trace = False, **kwargs):
        """Instantiate a regression model.
        
        :Arguments:

            * data : pandas.DataFrame
                data containing 'rt', 'response', column and any
                covariates you might want to use.
            * models : str or list of str
                Patsy linear model specifier.
                E.g. 'v ~ cov'
                You can include multiple linear models that influence
                separate DDM parameters.

        :Optional:

            * group_only_regressors : bool (default = True)
                Do not estimate individual subject parameters for all regressors.
            * keep_regressor_trace : bool (default = False)
                Whether to keep a trace of the regressor. This will use much more space,
                but needed for posterior predictive checks.
            * Additional keyword args are passed on to HDDM.

        :Note:

            Internally, HDDMnnRegressor uses patsy which allows for
            simple yet powerful model specification. For more information see:
            http://patsy.readthedocs.org/en/latest/overview.html

        :Example:

            Consider you have a trial-by-trial brain measure
            (e.g. fMRI) as an extra column called 'BOLD' in your data
            frame. You want to estimate whether BOLD has an effect on
            drift-rate. The corresponding model might look like
            this:
                ```python
                HDDMnnRegressor(data, 'v ~ BOLD')
                ```

            This will estimate an v_Intercept and v_BOLD. If v_BOLD is
            positive it means that there is a positive correlation
            between BOLD and drift-rate on a trial-by-trial basis.

            This type of mechanism also allows within-subject
            effects. If you have two conditions, 'cond1' and 'cond2'
            in the 'conditions' column of your data you may
            specify:
                ```python
                HDDMnnRegressor(data, 'v ~ C(condition)')
                ```
            This will lead to estimation of 'v_Intercept' for cond1
            and v_C(condition)[T.cond2] for cond1 + cond2.

        """
        kwargs['nn'] = True
        self.w_outlier = kwargs.pop('w_outlier', 0.1)
        self.network_type = kwargs.pop('network_type', 'mlp')
        self.network = None
        self.keep_regressor_trace = keep_regressor_trace
        if isinstance(models, (str, dict)):
            models = [models]
        
        group_only_nodes = list(kwargs.get('group_only_nodes', ()))
        self.reg_outcomes = set() # holds all the parameters that are going to be modeled as outcomes
        self.model = deepcopy(model)
        
        # Initialize data-structure that contains model descriptors
        self.model_descrs = []

        # Cycle through models, generate descriptors an add them to self.model_descrs
        for model in models:
            if isinstance(model, dict):
                try:
                    model_str = model['model']
                    link_func = model['link_func']
                except KeyError:
                    raise KeyError("HDDMnnRegressor requires a model specification either like {'model': 'v ~ 1 + C(your_variable)', 'link_func' lambda x: np.exp(x)} or just a model string")
            else:
                model_str = model
                link_func = lambda x: x

            # Split the model string to separate out dependent from independent variables
            separator = model_str.find('~')
            assert separator != -1, 'No outcome variable specified.'
            outcome = model_str[:separator].strip(' ')
            model_stripped = model_str[(separator + 1):]
            covariates = dmatrix(model_stripped, data).design_info.column_names # this uses Patsy to get a data matrix back

            print('outcomes: ', outcome)
            # Build model descriptor
            model_descr = {'outcome': outcome,
                           'model': model_stripped,
                           'params': ['{out}_{reg}'.format(out=outcome, reg=reg) for reg in covariates],
                           'link_func': link_func
            }
            
            self.model_descrs.append(model_descr)

            print("Adding these covariates:")
            print(model_descr['params'])

            if group_only_regressors:
                group_only_nodes += model_descr['params']
                kwargs['group_only_nodes'] = group_only_nodes
            self.reg_outcomes.add(outcome)

        if self.network_type == 'mlp':
            self.network = load_mlp(model = self.model)
            network_dict = {'network': self.network}
            #likelihood_ = hddm.likelihoods_mlp.make_mlp_likelihoods(model = self.model)

        # self.wfpt_nn = stochastic_from_dist('Wiennernn' + '_' + self.model,
        #                                     partial(likelihood_, **network_dict))
        self.wfpt_reg_class = generate_wfpt_nn_ddm_reg_stochastic_class(sampling_method = 'drift', model = self.model, **network_dict)
       
        # Attach the likelihood !
        # if self.model == 'ddm':
        #     self.wfpt_reg_class = generate_wfpt_nn_ddm_reg_stochastic_class(sampling_method = 'drift', model = 'ddm')
        # if self.model == 'angle':
        #     self.wfpt_reg_class = generate_wfpt_nn_angle_reg_stochastic_class(sampling_method = 'drift')
        # if self.model == 'ornstein':
        #     self.wfpt_reg_class = generate_wfpt_nn_ornstein_reg_stochastic_class(sampling_method = 'drift')
        # if self.model == 'levy':
        #     self.wfpt_reg_class = generate_wfpt_nn_levy_reg_stochastic_class(sampling_method = 'drift')
        # if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull2':
        #     self.wfpt_reg_class = generate_wfpt_nn_weibull_reg_stochastic_class(sampling_method = 'drift')
        # if self.model == 'full_ddm' or self.model == 'full_ddm2':
        #     self.wfpt_reg_class = generate_wfpt_nn_full_ddm_reg_stochastic_class(sampling_method = 'drift')

        super(HDDMnnRegressor, self).__init__(data, **kwargs)

        # Sanity checks
        for model_descr in self.model_descrs:
            for param in model_descr['params']:
                assert len(self.depends[param]) == 0, "When using patsy, you can not use any model parameter in depends_on."

    def __getstate__(self):
        d = super(HDDMnnRegressor, self).__getstate__()
        del d['wfpt_reg_class']
        for model in d['model_descrs']:
            if 'link_func' in model:
                print("WARNING: Will not save custom link functions.")
                del model['link_func']
        return d

    def __setstate__(self, d):
        d['wfpt_reg_class'] = deepcopy(wfpt_reg_like)
        print("WARNING: Custom link functions will not be loaded.")
        for model in d['model_descrs']:
            model['link_func'] = lambda x: x
        super(HDDMnnRegressor, self).__setstate__(d)

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
            wfpt_parents['alpha'] = knodes['alpha_bottom'] if 'alpha' in self.include else 1.5
        
        if self.model == 'angle':
            wfpt_parents['theta'] = knodes['theta_bottom'] if 'theta' in self.include else 0

        if self.model == 'full_ddm' or self.model == 'full_ddm2':
            wfpt_parents['sv'] = knodes['sv_bottom'] if 'sv' in self.include else 0 #self.default_intervars['sv']
            wfpt_parents['sz'] = knodes['sz_bottom'] if 'sz' in self.include else 0 #self.default_intervars['sz']
            wfpt_parents['st'] = knodes['st_bottom'] if 'st' in self.include else 0 #self.default_intervars['st']
       

        print('wfpt parents: ')
        print(wfpt_parents)

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)

        return Knode(self.wfpt_reg_class,
                     'wfpt',
                     observed = True,
                     col_name = ['response', 'rt'],
                     reg_outcomes = self.reg_outcomes,
                     **wfpt_parents)

    # TD def _create_stochastic_knodes_base(self, include:)
    def _create_stochastic_knodes_basic(self, include):
        knodes = OrderedDict()

        # PARAMETERS TO ACCOMODATE TRAINED PARAMETER BOUNDS BY MODEL
        if 'p_outlier' in include:
            knodes.update(self._create_family_invlogit('p_outlier',
                                                        value = 0.2,
                                                        g_tau = 10**-2,
                                                        std_std = 0.5
                                                        ))

        # SPLIT BY MODEL TO ACCOMMODATE TRAINED PARAMETER BOUNDS BY MODEL
        if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull2':
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
                                                               value = .5,
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

    def _create_stochastic_knodes(self, include):
        # Create all stochastic knodes except for the ones that we want to replace
        # with regressors. '.difference' makes that happen
        #knodes = self._create_stochastic_knodes_nn(include.difference(self.reg_outcomes))

        print('Printing reg outcome:')
        print(self.reg_outcomes)
        #include = set(include) # TD: Check why here include is not coming in as a set // This worked in hddm_nn.py
        includes_remainder = set(include).difference(self.reg_outcomes)
        print(includes_remainder)
        knodes = self._create_stochastic_knodes_basic(includes_remainder)

        print('knodes')
        print(knodes)
        
        # This is in dire need of refactoring. Like any monster, it just grew over time.
        # The main problem is that it's not always clear which prior to use. For the intercept
        # we want to use the original parameters' prior. Also for categoricals that do not
        # have an intercept, but not when the categorical is part of an interaction....

        # Create regressor params
        for reg in self.model_descrs:
            print('reg: ', reg)
            reg_parents = {}
            # Find intercept parameter
            intercept = np.asarray([param.find('Intercept') for param in reg['params']]) != -1
            
            # If no intercept specified (via 0 + C()) assume all C() are different conditions
            # -> all are intercepts
            
            if not np.any(intercept):
                # Has categorical but no interaction
                intercept = np.asarray([(param.find('C(') != -1) and (param.find(':') == -1)
                                        for param in reg['params']])

            for inter, param in zip(intercept, reg['params']):
                if inter:
                    # Intercept parameter should have original prior (not centered on 0)
                    param_lookup = param[:param.find('_')]
                    print('param_lookup passed to _create stochastic_knodes')
                    print(param_lookup)
                    reg_family = self._create_stochastic_knodes_basic([param_lookup])
                    
                    # Rename nodes to avoid collissions
                    names = list(reg_family.keys())
                    print(names)
                    for name in names:
                        print('name: ', name)
                        print('names: ', names)
                        knode = reg_family.pop(name)
                        knode.name = knode.name.replace(param_lookup, param, 1)
                        reg_family[name.replace(param_lookup, param, 1)] = knode
                    param_lookup = param

                else:
                    #param_lookup = param[:param.find('_')]
                    # This potentially needs change, we should cleverly constrain the covariate betas here
                    # right now the solution is to return  -np.inf in the likelihood if the betas are too big / small
                    # but this is sort of an expensive brute force procedure !
                    reg_family = self._create_family_normal(param)
                    param_lookup = param

                reg_parents[param] = reg_family['%s_bottom' % param_lookup]
                
                if reg not in self.group_only_nodes:
                    reg_family['%s_subj_reg' % param] = reg_family.pop('%s_bottom' % param_lookup)
                
                knodes.update(reg_family)
                self.slice_widths[param] = .05


            reg_knode = KnodeRegress(pm.Deterministic, "%s_reg" % reg['outcome'],
                                     regressor = reg,
                                     subj = self.is_group_model,
                                     plot = False,
                                     trace = False,
                                     hidden = True,
                                     keep_regressor_trace = self.keep_regressor_trace,
                                     **reg_parents)

            knodes['%s_bottom' % reg['outcome']] = reg_knode
            print(reg_knode)
        return knodes
