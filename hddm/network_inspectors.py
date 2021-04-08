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
        keras.model.predict_on_batch
        Returns a function that gives you access to a forward pass through the MLP. 
        This in turn expects as input a 2d np.array of datatype np.float32. Each row is filled with
        model parameters trailed by a reaction time and a choice.
        (e.g. input dims for a ddm MLP could be (3, 6), 3 datapoints and 4 parameters + reaction time and choice).
        Predict on batch then returns for each row of the input the log likelihood of the respective parameter vector and datapoint.

    :Example:
        >>> forward = hddm.network_inspectors.get_mlp(model = 'ddm')
        >>> data = np.array([[0.5, 1.5, 0.5, 0.5, 1.0, -1.0], [0.5, 1.5, 0.5, 0.5, 1.0, -1.0]], dtype = np.float32)
        >>> forward(data)
    """

    network = load_mlp(model = model)
    return network.predict_on_batch

def get_cnn(model = 'angle', nbin = 512):
    """ Returns tensorflow CNN which is the basis of the CNN likelihoods

    :Arguments:
        model: str <default='angle'>
        Specifies the models you would like to load
    
    Returns:
        function
            Returns a function that you can call passing as an argument a 1d or 2d np.array with datatype np.float32.
            The shape of the input to this function should match the number of parameter vectors (rows) and the corresponding parameters (cols).
            Per paraemter vector passed, this function will give out an np.array() of shape (1, n_choice_options * nbins).
            This output defines a probability mass functions over discretized rt / choice space. The first 'n_choice_options' indices
            define the probability of landing in the first bin for each choice option etc..

    Example:   
        :Example:
        >>> forward = hddm.network_inspectors.get_cnn(model = 'ddm')
        >>> data = np.array([[0.5, 1.5, 0.5, 0.5], [0.5, 1.5, 0.5, 0.5]], dtype = np.float32)
        >>> forward(data)        
    """
    network = load_cnn(model = model, nbin = nbin)
    return network