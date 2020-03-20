
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_adhd


class HDDMadhd(HDDM):
    """

    """

    def __init__(self, *args, **kwargs):
        self.b_v_zpos = kwargs.pop('b_v_zpos', False)
        self.b_v_zneg = kwargs.pop('b_v_zneg', False)
        self.b_v_dmed = kwargs.pop('b_v_dmed', False)
        self.b_v_ztrain = kwargs.pop('b_v_ztrain', False)
        self.b_v_zarm = kwargs.pop('b_v_zarm', False)
        self.b_v_zpos_dmed = kwargs.pop('b_v_zpos_dmed', False)
        self.b_v_zneg_dmed = kwargs.pop('b_v_zneg_dmed', False)
        self.b_v_zpos_ztrain = kwargs.pop('b_v_zpos_ztrain', False)
        self.b_v_zneg_ztrain = kwargs.pop('b_v_zneg_ztrain', False)
        self.b_v_zpos_zarm = kwargs.pop('b_v_zpos_zarm', False)
        self.b_v_zneg_zarm = kwargs.pop('b_v_zneg_zarm', False)
        self.b_v_dmed_ztrain = kwargs.pop('b_v_dmed_ztrain', False)
        self.b_v_zpos_dmed_ztrain = kwargs.pop('b_v_zpos_dmed_ztrain', False)
        self.b_v_zneg_dmed_ztrain = kwargs.pop('b_v_zneg_dmed_ztrain', False)

        self.b_a_dmed = kwargs.pop('b_a_dmed', False)
        self.b_a_ztrain = kwargs.pop('b_a_ztrain', False)
        self.b_a_zarm = kwargs.pop('b_a_zarm', False)
        self.b_a_dmed_ztrain = kwargs.pop('b_a_dmed_ztrain', False)

        self.b_t_dmed = kwargs.pop('b_t_dmed', False)
        self.b_t_ztrain = kwargs.pop('b_t_ztrain', False)
        self.b_t_zarm = kwargs.pop('b_t_zarm', False)
        self.b_t_dmed_ztrain = kwargs.pop('b_t_dmed_ztrain', False)

        self.b_z_dmed = kwargs.pop('b_z_dmed', False)
        self.b_z_ztrain = kwargs.pop('b_z_ztrain', False)
        self.b_z_zarm = kwargs.pop('b_z_zarm', False)
        self.b_z_dmed_ztrain = kwargs.pop('b_z_dmed_ztrain', False)

        self.wfpt_adhd_class = WienerADHD

        super(HDDMadhd, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMadhd, self)._create_stochastic_knodes(include)
        if self.b_v_zpos:
            knodes.update(self._create_family_normal(
                    'b_v_zpos', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg:
            knodes.update(self._create_family_normal(
                    'b_v_zneg', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_ztrain:
            knodes['b_v_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_ztrain'])
        if self.b_v_zarm:
            knodes['b_v_zarm_bottom'] = Knode(pymc.Normal, 'b_v_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zarm'])
        if self.b_v_zpos_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_zpos_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_zneg_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zpos_ztrain:
            knodes['b_v_zpos_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_zpos_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_ztrain'])
        if self.b_v_zneg_ztrain:
            knodes['b_v_zneg_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_zneg_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_ztrain'])
        if self.b_v_zpos_zarm:
            knodes['b_v_zpos_zarm_bottom'] = Knode(pymc.Normal, 'b_v_zpos_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_zarm'])
        if self.b_v_zneg_zarm:
            knodes['b_v_zneg_zarm_bottom'] = Knode(pymc.Normal, 'b_v_zneg_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_zarm'])
        if self.b_v_dmed_ztrain:
            knodes['b_v_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_dmed_ztrain'])
        if self.b_v_zpos_dmed_ztrain:
            knodes['b_v_zpos_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_zpos_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_dmed_ztrain'])
        if self.b_v_zneg_dmed_ztrain:
            knodes['b_v_zneg_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_v_zneg_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_dmed_ztrain'])
        if self.b_a_dmed:
            knodes.update(self._create_family_normal(
                    'b_a_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_a_ztrain:
            knodes['b_a_ztrain_bottom'] = Knode(pymc.Normal, 'b_a_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_ztrain'])
        if self.b_a_zarm:
            knodes['b_a_zarm_bottom'] = Knode(pymc.Normal, 'b_a_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_zarm'])
        if self.b_a_dmed_ztrain:
            knodes['b_a_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_a_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_dmed_ztrain'])
        if self.b_t_dmed:
            knodes.update(self._create_family_normal(
                    'b_t_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_t_ztrain:
            knodes['b_t_ztrain_bottom'] = Knode(pymc.Normal, 'b_t_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_ztrain'])
        if self.b_t_zarm:
            knodes['b_t_zarm_bottom'] = Knode(pymc.Normal, 'b_t_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_zarm'])
        if self.b_t_dmed_ztrain:
            knodes['b_t_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_t_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_dmed_ztrain'])
        if self.b_z_dmed:
            knodes.update(self._create_family_normal(
                    'b_z_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_z_ztrain:
            knodes['b_z_ztrain_bottom'] = Knode(pymc.Normal, 'b_z_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_ztrain'])
        if self.b_z_zarm:
            knodes['b_z_zarm_bottom'] = Knode(pymc.Normal, 'b_z_zarm', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_zarm'])
        if self.b_z_dmed_ztrain:
            knodes['b_z_dmed_ztrain_bottom'] = Knode(pymc.Normal, 'b_z_dmed_ztrain', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_dmed_ztrain'])
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMadhd, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['b_v_zpos'] = knodes['b_v_zpos_bottom'] if self.b_v_zpos else 0.0
        wfpt_parents['b_v_zneg'] = knodes['b_v_zneg_bottom'] if self.b_v_zneg else 0.0
        wfpt_parents['b_v_dmed'] = knodes['b_v_dmed_bottom'] if self.b_v_dmed else 0.0
        wfpt_parents['b_v_ztrain'] = knodes['b_v_ztrain_bottom'] if self.b_v_ztrain else 0.0
        wfpt_parents['b_v_zarm'] = knodes['b_v_zarm_bottom'] if self.b_v_zarm else 0.0
        wfpt_parents['b_v_zpos_dmed'] = knodes['b_v_zpos_dmed_bottom'] if self.b_v_zpos_dmed else 0.0
        wfpt_parents['b_v_zneg_dmed'] = knodes['b_v_zneg_dmed_bottom'] if self.b_v_zneg_dmed else 0.0
        wfpt_parents['b_v_zpos_ztrain'] = knodes['b_v_zpos_ztrain_bottom'] if self.b_v_zpos_ztrain else 0.0
        wfpt_parents['b_v_zneg_ztrain'] = knodes['b_v_zneg_ztrain_bottom'] if self.b_v_zneg_ztrain else 0.0
        wfpt_parents['b_v_zpos_zarm'] = knodes['b_v_zpos_zarm_bottom'] if self.b_v_zpos_zarm else 0.0
        wfpt_parents['b_v_zneg_zarm'] = knodes['b_v_zneg_zarm_bottom'] if self.b_v_zneg_zarm else 0.0
        wfpt_parents['b_v_dmed_ztrain'] = knodes['b_v_dmed_ztrain_bottom'] if self.b_v_dmed_ztrain else 0.0
        wfpt_parents['b_v_zpos_dmed_ztrain'] = knodes['b_v_zpos_dmed_ztrain_bottom'] if self.b_v_zpos_dmed_ztrain else 0.0
        wfpt_parents['b_v_zneg_dmed_ztrain'] = knodes['b_v_zneg_dmed_ztrain_bottom'] if self.b_v_zneg_dmed_ztrain else 0.0
        wfpt_parents['b_a_dmed'] = knodes['b_a_dmed_bottom']  if self.b_a_dmed else 0.0
        wfpt_parents['b_a_ztrain'] = knodes['b_a_ztrain_bottom'] if self.b_a_ztrain else 0.0
        wfpt_parents['b_a_zarm'] = knodes['b_a_zarm_bottom'] if self.b_a_zarm else 0.0
        wfpt_parents['b_a_dmed_ztrain'] = knodes['b_a_dmed_ztrain_bottom'] if self.b_a_dmed_ztrain else 0.0
        wfpt_parents['b_t_dmed'] = knodes['b_t_dmed_bottom']  if self.b_t_dmed else 0.0
        wfpt_parents['b_t_ztrain'] = knodes['b_t_ztrain_bottom'] if self.b_t_ztrain else 0.0
        wfpt_parents['b_t_zarm'] = knodes['b_t_zarm_bottom'] if self.b_t_zarm else 0.0
        wfpt_parents['b_t_dmed_ztrain'] = knodes['b_t_dmed_ztrain_bottom'] if self.b_t_dmed_ztrain else 0.0
        wfpt_parents['b_z_dmed'] = knodes['b_z_dmed_bottom'] if self.b_z_dmed else 0.0
        wfpt_parents['b_z_zarm'] = knodes['b_z_zarm_bottom'] if self.b_z_zarm else 0.0
        wfpt_parents['b_z_ztrain'] = knodes['b_z_ztrain_bottom'] if self.b_z_ztrain else 0.0
        wfpt_parents['b_z_dmed_ztrain'] = knodes['b_z_dmed_ztrain_bottom'] if self.b_z_dmed_ztrain else 0.0
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_adhd_class, 'wfpt', observed=True, col_name=['rt', 'zpos','zneg','dmed','dsess','ztrain','zarm'], **wfpt_parents)


def wienerADHD_like(x, v, sv, a, z, sz, t, st,b_v_zpos, b_v_zneg, b_v_dmed, b_v_ztrain,b_v_zarm,b_v_zpos_zarm,b_v_zneg_zarm, b_v_zpos_dmed, b_v_zneg_dmed, b_v_zpos_ztrain, b_v_zneg_ztrain, b_v_dmed_ztrain, b_v_zpos_dmed_ztrain, b_v_zneg_dmed_ztrain, b_a_dmed, b_a_ztrain,b_a_zarm, b_a_dmed_ztrain, b_t_dmed, b_t_ztrain, b_t_zarm, b_t_dmed_ztrain, b_z_dmed, b_z_ztrain, b_z_zarm, b_z_dmed_ztrain, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params
    zpos = x['zpos'].values.astype(float)
    zneg = x['zneg'].values.astype(float)
    dmed = x['dmed'].values.astype(float)
    dsess = x['dsess'].values.astype(float)
    ztrain = x['ztrain'].values.astype(float)
    zarm = x['zarm'].values.astype(float)
    return wiener_like_adhd(x['rt'].values, zpos, zneg, dmed, dsess, ztrain,zarm ,b_v_zpos, b_v_zneg, b_v_dmed, b_v_ztrain,b_v_zarm,b_v_zpos_zarm,b_v_zneg_zarm, b_v_zpos_dmed, b_v_zneg_dmed, b_v_zpos_ztrain, b_v_zneg_ztrain, b_v_dmed_ztrain, b_v_zpos_dmed_ztrain, b_v_zneg_dmed_ztrain, b_a_dmed, b_a_ztrain,b_a_zarm, b_a_dmed_ztrain, b_t_dmed, b_t_ztrain, b_t_zarm, b_t_dmed_ztrain, b_z_dmed, b_z_ztrain, b_z_zarm, b_z_dmed_ztrain, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
WienerADHD = stochastic_from_dist('wienerADHD', wienerADHD_like)



