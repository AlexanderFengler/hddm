
import hddm.simulators
#from hddm.simulators import boundary_functions
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import hddm
import sys
import kabuki
import pandas as pd
import string
import argparse
from kabuki.analyze import post_pred_gen, post_pred_compare_stats
from hddm.keras_models import load_mlp
from hddm.cnn.wrapper import load_cnn
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import truncnorm
from scipy.stats import scoreatpercentile
from scipy.stats.mstats import mquantiles
from copy import deepcopy

def run_simulator():
    return basic_simulator([0, 1, 0.5, 0.3],
                            model = 'angle',
                            n_samples = 10000,
                            n_trials = 10,
                            delta_t = 0.001,
                            max_t = 20,
                            cartoon = False,
                            bin_dim = None, 
                            bin_pointwise = False)


