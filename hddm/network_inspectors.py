from hddm.simulators import *
#from hddm.simulators import boundary_functions
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import hddm
import sys
import kabuki
import pandas as pd
import seaborn as sns
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


def get_mlp(model = 'angle'):
    """ Returns the keras network which is the basis of the MLP likelihoods

    :Arguments:
        model: str <default='angle'>
        Specifies the models you would like to load
    Returns:
        keras.model: You can call a forward pass through the model via .predict_on_batch()
    """
    network = load_mlp(model = model)
    return network.predict_on_batch

def get_cnn(model = 'angle', nbin = 512):
    """ Returns tensorflow CNN which is the basis of the CNN likelihoods

    :Arguments:
        model: str <default='angle'>
        Specifies the models you would like to load
    Returns:
        function: 
            Returns a function that you can call passing as an argument a 1d or 2d np.array with datatype np.float32.
            The shape of the input to this function should match the number of parameter vectors (rows) and the corresponding parameters (cols).
                
    """
    network = load_cnn(model = model, nbin = nbin)
    return network