from hddm.simulators import *
#from hddm.simulators import boundary_functions
import numpy as np

#plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm

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
from hddm.simulators.basic_simulator import *
import scipy as scp
from sklearn.neighbors import KernelDensity
import os

model_config = hddm.simulators.model_config

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

class logkde():
    def __init__(self,
                 simulator_data, # Simulator_data is the kind of data returned by the simulators in ddm_data_simulatoin.py
                 bandwidth_type = 'silverman',
                 auto_bandwidth = True):

        self.attach_data_from_simulator(simulator_data)
        self.generate_base_kdes(auto_bandwidth = auto_bandwidth,
                                bandwidth_type = bandwidth_type)
        self.simulator_info = simulator_data[2]

    # Function to compute bandwidth parameters given data-set
    # (At this point using Silverman rule)
    def compute_bandwidths(self,
                           type = 'silverman'):

        self.bandwidths = []
        if type == 'silverman':
            for i in range(0, len(self.data['choices']), 1):
                if len(self.data['rts'][i]) == 0:
                    self.bandwidths.append('no_base_data')
                else:
                    bandwidth_tmp = bandwidth_silverman(sample = np.log(self.data['rts'][i]))
                    if bandwidth_tmp > 0:
                        self.bandwidths.append(bandwidth_tmp)
                    else:
                        #print(self.data['rts'][i])
                        self.bandwidths.append('no_base_data')

    # Function to generate basic kdes
    # I call the function generate_base_kdes because in the final evaluation computations
    # we adjust the input and output of the kdes appropriately (we do not use them directly)
    def generate_base_kdes(self,
                           auto_bandwidth = True,
                           bandwidth_type  = 'silverman'):

        # Compute bandwidth parameters
        if auto_bandwidth:
            self.compute_bandwidths(type = bandwidth_type)

        # Generate the kdes
        self.base_kdes = []
        for i in range(0, len(self.data['choices']), 1):
            if self.bandwidths[i] == 'no_base_data':
                self.base_kdes.append('no_base_data')
                #print('no_base_data reported')
            else: 
                self.base_kdes.append(KernelDensity(kernel = 'gaussian',
                                                    bandwidth = self.bandwidths[i]).fit(np.log(self.data['rts'][i])))

    # Function to evaluate the kde log likelihood at chosen points
    def kde_eval(self,
                 data = ([], []),  #kde
                 log_eval = True):
        
        # Initializations
        log_rts = np.log(data[0])
        log_kde_eval = np.log(data[0])
        choices = np.unique(data[1])
        #print('choices to iterate:', choices)
        #print('choices from kde:', self.data['choices'])
        
        # Main loop
        for c in choices:
            
            # Get data indices where choice == c
            choice_idx_tmp = np.where(data[1] == c)
            
            # Main step: Evaluate likelihood for rts corresponding to choice == c
            if self.base_kdes[self.data['choices'].index(c)] == 'no_base_data':
                log_kde_eval[choice_idx_tmp] = -66.77497 # the number corresponds to log(1e-29) # --> log(1 / n) + log(1 / 20)
            else:
                log_kde_eval[choice_idx_tmp] = np.log(self.data['choice_proportions'][self.data['choices'].index(c)]) + \
                self.base_kdes[self.data['choices'].index(c)].score_samples(np.expand_dims(log_rts[choice_idx_tmp], 1)) - \
                log_rts[choice_idx_tmp]
            
        if log_eval == True:
            return log_kde_eval
        else:
            return np.exp(log_kde_eval)
    
    def kde_sample(self,
                   n_samples = 2000,
                   use_empirical_choice_p = True,
                   alternate_choice_p = 0):
        
        # sorting the which list in ascending order 
        # this implies that we return the kde_samples array so that the
        # indices reflect 'choice-labels' as provided in 'which' in ascending order
        kde_samples = []
        
        rts = np.zeros((n_samples, 1))
        choices = np.zeros((n_samples, 1))
        
        n_by_choice = []
        for i in range(0, len(self.data['choices']), 1):
            if use_empirical_choice_p == True:
                n_by_choice.append(round(n_samples * self.data['choice_proportions'][i]))
            else:
                n_by_choice.append(round(n_samples * alternate_choice_p[i]))
        
        # Catch a potential dimension error if we ended up rounding up twice
        if sum(n_by_choice) > n_samples: 
            n_by_choice[np.argmax(n_by_choice)] -= 1
        elif sum(n_by_choice) < n_samples:
            n_by_choice[np.argmax(n_by_choice)] += 1
            #print('rounding error catched')
            choices[n_samples - 1, 0] = np.random.choice(self.data['choices'])
            #print('resolution: ', choices[n_samples - 1, 0])
            #print('choices allowed: ', self.data['choices'])
            
        # Get samples
        cnt_low = 0
        for i in range(0, len(self.data['choices']), 1):
            if n_by_choice[i] > 0:
                #print('sum of n_by_choice:', sum(n_by_choice))
                cnt_high = cnt_low + n_by_choice[i]
                
                if self.base_kdes[i] != 'no_base_data':
                    rts[cnt_low:cnt_high] = np.exp(self.base_kdes[i].sample(n_samples = n_by_choice[i]))
                else:
                    rts[cnt_low:cnt_high, 0] = np.random.uniform(low = 0, high = 20, size = n_by_choice[i])
                
                choices[cnt_low:cnt_high, 0] = np.repeat(self.data['choices'][i], n_by_choice[i])
                cnt_low = cnt_high
                
        return ((rts, choices, self.simulator_info)) 
        
    # Helper function to transform ddm simulator output to dataset suitable for
    # the kde function class
    def attach_data_from_simulator(self,
                                   simulator_data = ([0, 2, 4], [-1, 1, -1])):

        choices = np.unique(simulator_data[2]['possible_choices'])
        
        n = len(simulator_data[0])
        self.data = {'rts': [],
                     'choices': [],
                     'choice_proportions': []}

        # Loop through the choices made to get proportions and separated out rts
        for c in choices:
            self.data['choices'].append(c)
            rts_tmp = np.expand_dims(simulator_data[0][simulator_data[1] == c], axis = 1)
            prop_tmp = len(rts_tmp) / n
            self.data['rts'].append(rts_tmp)
            self.data['choice_proportions'].append(prop_tmp)
            

# Support functions (accessible from outside the main class defined in script)
def bandwidth_silverman(sample = [0,0,0], 
                        std_cutoff = 1e-3, 
                        std_proc = 'restrict', # options 'kill', 'restrict'
                        std_n_1 = 1e-1 # HERE WE CAN ALLOW FOR SOMETHING MORE INTELLIGENT
                       ): 
    
    # Compute sample std and number of samples
    std = np.std(sample)
    n = len(sample)
    
    # Deal with very small stds and n = 1 case
    if n > 1:
        if std < std_cutoff:
            if std_proc == 'restrict':
                std = std_cutoff
            if std_proc == 'kill':
                std = 0
    else:
        std = std_n_1
    
    return np.power((4/3), 1/5) * std * np.power(n, (-1/5))

def kde_vs_mlp_likelihoods(#ax_titles = [],
                           parameter_df = [],
                           cols = 3,
                           model = 'angle',
                           n_samples = 10,
                           nreps = 10,
                           save = True,
                           show = False):
    
    # Get prediction from navarro if traindatanalytic = 1
    # if traindatanalytic:
    #     ll_out_gt = cdw.batch_fptd(plot_data[:, 0] * plot_data[:, 1], 
    #                            v = parameter_matrix[i, 0],
    #                            a = parameter_matrix[i, 1],
    #                            w = parameter_matrix[i, 2],
    #                            ndt = parameter_matrix[i, 3])

    #     sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
    #              ll_out_gt,
    #              color = 'black',
    #              alpha = 0.5,
    #              label = 'TRUE',
    #              ax = ax[row_tmp, col_tmp])
    
    # Get predictions from simulations /kde
    
    mpl.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    
    # Initialize rows and graph parameters
    rows = int(np.ceil(parameter_df.shape[0] / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (10, 10), 
                           sharex = True, 
                           sharey = False)
    
    fig.suptitle('Likelihoods KDE - MLP' + ': ' + model.upper().replace('_', '-'), fontsize = 40)
    sns.despine(right = True)
    
    # Data template
    plot_data = np.zeros((4000, 2))
    plot_data[:, 0] = np.concatenate(([i * 0.0025 for i in range(2000, 0, -1)], [i * 0.0025 for i in range(1, 2001, 1)]))
    plot_data[:, 1] = np.concatenate((np.repeat(-1, 2000), np.repeat(1, 2000)))
    
    # Load Keras model and initialize batch container
    keras_model = get_mlp(model = model)
    #keras_model = keras.models.load_model(network_dir + 'model_final.h5')
    keras_input_batch = np.zeros((4000, parameter_df.shape[1] + 2))
    keras_input_batch[:, parameter_df.shape[1]:] = plot_data

    for i in range(parameter_df.shape[0]):
        
        print('Making Plot: ', i)
        
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        # Get predictions from keras model
        keras_input_batch[:, :parameter_df.shape[1]] = parameter_df.iloc[i, :].values
        ll_out_keras = keras_model.predict(keras_input_batch, 
                                           batch_size = 100)
        
        if not traindatanalytic:
            for j in range(nreps):
                out = simulator(theta = parameter_df.iloc[i, :].values,
                                model = model,
                                n_samples = n_samples,
                                max_t = 20,
                                delta_t = 0.001)

                mykde = logkde((out[0], out[1], out[2]))
                ll_out_gt = mykde.kde_eval((plot_data[:, 0], plot_data[:, 1]))

                # Plot kde predictions
                if j == 0:
                    sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                                 np.exp(ll_out_gt),
                                 color = 'black',
                                 alpha = 0.5,
                                 label = 'KDE',
                                 ax = ax[row_tmp, col_tmp])
                elif j > 0:
                    sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                                 np.exp(ll_out_gt),
                                 color = 'black',
                                 alpha = 0.5,
                                 ax = ax[row_tmp, col_tmp])

            # Plot keras predictions
            sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                         np.exp(ll_out_keras[:, 0]),
                         color = 'green',
                         label = 'MLP',
                         alpha = 1,
                         ax = ax[row_tmp, col_tmp])

        # Legend adjustments
        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(loc = 'upper left', 
                                        fancybox = True, 
                                        shadow = True,
                                        fontsize = 12)
        else: 
            ax[row_tmp, col_tmp].legend().set_visible(False)
        
        
        if row_tmp == rows - 1:
            ax[row_tmp, col_tmp].set_xlabel('rt', 
                                            fontsize = 24);
        else:
            ax[row_tmp, col_tmp].tick_params(color = 'white')
        
        if col_tmp == 0:
            ax[row_tmp, col_tmp].set_ylabel('likelihood', 
                                            fontsize = 24);
        
        # tmp title


        ax[row_tmp, col_tmp].set_title(ax_titles[i],
                                       fontsize = 20)
        ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 16)
        ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 16)
        
    for i in range(len(ax_titles), rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.subplots_adjust(top = 0.9)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
    if save == True:
        if os.path.isdir('figures/'):
            pass
        else:
            os.mkdir('figures/')

        plt.savefig('figures/' + 'kde_vs_mlp_plot' + '.png',
                    format = 'png',
                    transparent = True,
                    frameon = False)
    
    if show:
        plt.show()

    plt.close()

    return
   