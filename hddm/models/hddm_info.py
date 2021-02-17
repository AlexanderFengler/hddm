from collections import OrderedDict
import inspect
import numpy as np
import pymc as pm
import kabuki.step_methods as steps
from hddm.models import HDDMBase
from kabuki.hierarchical import Knode

class HDDM(HDDMBase):
    """Create hierarchical drift-diffusion model in which each subject
    has a set of parameters that are constrained by a group distribution.

    :Arguments:
        data : pandas.DataFrame
            Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
              * 'subj_idx': A unique ID (int) of each subject.
              * Other user-defined columns that can be used in depends_on
                keyword.

    :Optional:
        informative : bool <default=True>
            Whether to use informative priors (True) or vague priors
            (False).  Information about the priors can be found in the
            methods section.  If you run a classical DDM experiment
            you should use this. However, if you apply the DDM to a
            novel domain like saccade data where RTs are much lower,
            or RTs of rats, you should probably set this to False.

        is_group_model : bool
            If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.

        depends_on : dict
            Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)

            :Example:

                >>> hddm.HDDM(data, depends_on={'v': 'difficulty'})

                Separate drift-rate parameters will be estimated
                for each difficulty. Requires 'data' to have a
                column difficulty.

        bias : bool
            Whether to allow a bias to be estimated. This
            is normally used when the responses represent
            left/right and subjects could develop a bias towards
            responding right. This is normally never done,
            however, when the 'response' column codes
            correct/error.

        p_outlier : double (default=0)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.

        default_intervars : dict (default = {'sz': 0, 'st': 0, 'sv': 0})
            Fix intertrial variabilities to a certain value. Note that this will only
            have effect for variables not estimated from the data.

        plot_var : bool
             Plot group variability parameters when calling pymc.Matplot.plot()
             (i.e. variance of Normal distribution.)

        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)

        std_depends : bool (default=False)
             Should the depends_on keyword affect the group std node.
             If True it means that both, group mean and std will be split
             by condition.

        wiener_params : dict
             Parameters for wfpt evaluation and
             numerical integration.

         :Parameters:
             * err: Error bound for wfpt (default 1e-4)
             * n_st: Maximum depth for numerical integration for st (default 2)
             * n_sz: Maximum depth for numerical integration for Z (default 2)
             * use_adaptive: Whether to use adaptive numerical integration (default True)
             * simps_err: Error bound for Simpson integration (default 1e-3)

    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc.sample(5000, burn=20) # Sample from posterior

    """

    def __init__(self, *args, **kwargs):
        self.slice_widths = {'a': 1, 
                             't': 0.01, 
                             'a_std': 1,
                             't_std': 0.15, 
                             'sz': 1.1, 
                             'v': 1.5,
                             'st': 0.1, 
                             'sv': 0.5, # from sv = 3.00 
                             'z_trans': 0.2, 
                             'z': 0.1,
                             'p_outlier': 1., 
                             'v_std': 1, 
                             'alpha': 1., 
                             'dual_alpha': 1.5, 
                             'theta': 0.1,
                             'beta': 1.,
                             'g': 0.5}

        # Is this used at all ???
        self.emcee_dispersions = {'a': 1, 't': 0.1, 'a_std': 1, 't_std': 0.15, 'sz': 1.1, 'v': 1.5,
                                  'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                                  'p_outlier': 1., 'v_std': 1,'alpha': 1.5,'dual_alpha': 1.5,'theta': 0.1}

        self.is_informative = kwargs.pop('informative', True)

        super(HDDM, self).__init__(*args, **kwargs)

        print('Is Neural Net? ', self.nn)

    def _create_stochastic_knodes(self, include):
        if self.nn:
            if self.is_informative:
                return 'Informative Priors are not yet implementend for LANs, coming soon!'
            else:
                return self._create_stochastic_knodes_nn_noninfo(include)

        else:
            if self.is_informative:
                return self._create_stochastic_knodes_info(include)
            else:
                return self._create_stochastic_knodes_noninfo(include)


    def _create_stochastic_knodes_nn_noninfo(self, include):
        knodes = OrderedDict()
        
        # SPLIT BY MODEL TO ACCOMMODATE TRAINED PARAMETER BOUNDS BY MODEL

        # PARAMETERS COMMON TO ALL MODELS
        if 'p_outlier' in include:
            knodes.update(self._create_family_invlogit('p_outlier',
                                                        value = 0.2,
                                                        g_tau = 10**-2,
                                                        std_std = 0.5
                                                        ))

        if self.model == 'weibull' or self.model == 'weibull_cdf' or self.model == 'weibull_cdf2':
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
                                                           std_std = 0.5,
                                                           lower = 0.2,
                                                           upper = 0.8
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
                                                           std_std = 0.5,
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
                                                           std_std = 0.5,
                                                           lower = 0.1,
                                                           upper = 0.9
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
                                                           std_std = 0.5,
                                                           lower = 0.1,
                                                           upper = 0.9
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
                                                           std_std = 0.5,
                                                           lower = 0.3,
                                                           upper = 0.7
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
                                                           std_std = 0.5,
                                                           lower = 0.2,
                                                           upper = 0.8
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
                                                           std_std = 0.5,
                                                           lower = 0.2,
                                                           upper = 0.8
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
                                                           std_std = 0.5,
                                                           lower = 0.1,
                                                           upper = 0.9,
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


    def _create_stochastic_knodes_info(self, include):
        knodes = OrderedDict()
        if 'a' in include:
            knodes.update(self._create_family_gamma_gamma_hnormal('a', 
                                                                  g_mean = 1.5,
                                                                  g_std = 0.75, 
                                                                  std_std = 2,
                                                                  std_value = 0.1, 
                                                                  value = 1))
        if 'v' in include:
            knodes.update(self._create_family_normal_normal_hnormal('v', 
                                                                    value = 2,
                                                                    g_mu = 2, 
                                                                    g_tau = 3**-2, 
                                                                    std_std = 2))
        if 't' in include:
            knodes.update(self._create_family_gamma_gamma_hnormal('t', 
                                                                  g_mean = .4, 
                                                                  g_std = 0.2, 
                                                                  value = 0.001, 
                                                                  std_std = 1, 
                                                                  std_value = 0.2))
        if 'sv' in include:
            knodes['sv_bottom'] = Knode(pm.HalfNormal, 'sv', 
                                        tau = 2**-2, 
                                        value = 1, 
                                        depends = self.depends['sv'])
        if 'sz' in include:
            knodes['sz_bottom'] = Knode(pm.Beta, 'sz', 
                                        alpha = 1,
                                        beta = 3, 
                                        value = 0.01, 
                                        depends = self.depends['sz'])
        if 'st' in include:
            knodes['st_bottom'] = Knode(pm.HalfNormal, 'st', 
                                        tau = 0.3**-2, 
                                        value = 0.001, 
                                        depends = self.depends['st'])
        if 'z' in include:
            knodes.update(self._create_family_invlogit('z', 
                                                       value = .5, 
                                                       g_tau = 0.5**-2, 
                                                       std_std = 0.05))
        if 'p_outlier' in include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 'p_outlier', 
                                               alpha = 1, 
                                               beta = 15, 
                                               value = 0.01, 
                                               depends = self.depends['p_outlier'])

        return knodes

    def _create_stochastic_knodes_noninfo(self, include):
        knodes = OrderedDict()

        if 'a' in include:
            knodes.update(self._create_family_trunc_normal('a',
                                                           lower = 0.1,
                                                           upper = 6,
                                                           value = 1,
                                                           std_upper = 1 # added AF
                                                           ))
        if 'v' in include:
            knodes.update(self._create_family_trunc_normal('v', 
                                                           lower = - 3.0,
                                                           upper = 3.0,
                                                           value = 0,
                                                           std_upper = 1.5))
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
                                                       std_std = 0.5))

        # Below are parameters that are by default treated as global (no group distribution)
        if 'sv' in include:
            knodes['sv_bottom'] = Knode(pm.Uniform, 
                                        'sv', 
                                        lower = 1e-6,
                                        upper = 1e3,
                                        value = 1,
                                        depends = self.depends['sv'])
        if 'sz' in include:
            knodes['sz_bottom'] = Knode(pm.Beta, 
                                        'sz', 
                                        alpha = 1,
                                        beta = 1,
                                        value = 0.01,
                                        depends = self.depends['sz'])
        if 'st' in include:
            knodes['st_bottom'] = Knode(pm.Uniform, 
                                        'st', 
                                        lower = 1e-6,
                                        upper = 1e3,
                                        value = 0.01,
                                         depends = self.depends['st'])
        if 'p_outlier' in include:
            knodes['p_outlier_bottom'] = Knode(pm.Beta, 
                                               'p_outlier',
                                               alpha = 1,
                                               beta = 1,
                                               value = 0.01,
                                               depends = self.depends['p_outlier'])
        return knodes

    def pre_sample(self, use_slice = True):
        for name, node_descr in self.iter_stochastics():
            node = node_descr['node']
            if isinstance(node, pm.Normal) and np.all([isinstance(x, pm.Normal) for x in node.extended_children]):
                self.mc.use_step_method(steps.kNormalNormal,
                                        node)
            else:
                knode_name = node_descr['knode_name'].replace('_subj', '')
                if knode_name in ['st', 'sv', 'sz']:
                    left = 0
                else:
                    left = None
                self.mc.use_step_method(steps.SliceStep,
                                        node, 
                                        width = self.slice_widths.get(knode_name, 1),
                                        left = left,
                                        maxiter = 5000)

    def _create_an_average_model(self):
        """
        create an average model for group model quantiles optimization.
        """

        #this code only check that the arguments are as expected, i.e. the constructor was not change
        #since we wrote this function
        super_init_function = super(self.__class__, self).__init__
        init_args = set(inspect.getargspec(super_init_function).args)
        known_args = set(['wiener_params', 'include', 'self', 'bias', 'data', 'p_outlier'])
        assert known_args.issuperset(init_args), "Arguments of the constructor are not as expected"

        #create the avg model
        avg_model  = self.__class__(self.data,
                                    include = self.include, 
                                    is_group_model = False, 
                                    **self._kwargs)
        return avg_model
