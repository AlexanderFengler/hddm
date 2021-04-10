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

# def run_simulator():
#     return simulator([0, 1, 0.5, 0.3, 0.3],
#                             model = 'angle',
#                             n_samples = 10000,
#                             n_trials = 1,
#                             delta_t = 0.001,
#                             max_t = 20,
#                             cartoon = False,
#                             bin_dim = None, 
#                             bin_pointwise = False)


# Plot preprocessing functions
def _make_trace_plotready_single_subject(hddm_trace = None, model = ''):
    """Internal function to turn trace into data-format expected by plots. This
       version handles single subject data (a single subject's dataset).

    Arguments:
        hddm_trace: panda DataFrame
            Output of the hddm.get_traces() function.
        model: str <default=''>
            Name of the model that was fit (e.g. 'ddm', 'weibull', 'levy' ...)
    Return:
        panda DataFrame: Adjusted traces.
    """
    
    posterior_samples = np.zeros(hddm_trace.shape)
    
    cnt = 0
    for param in model_config[model]['params']:
        if param == 'z':
            posterior_samples[:, cnt] = model_config[model]['param_bounds'][0][model_config[model]['params'].index('z')] + \
                (  model_config[model]['param_bounds'][1][model_config[model]['params'].index('z')] - model_config[model]['param_bounds'][0][model_config[model]['params'].index('z')] ) * (1 / (1 + np.exp( - hddm_trace['z_trans'])))
            # posterior_samples[:, cnt] = 1 / (1 + np.exp( - hddm_trace['z_trans']))
        else:
            posterior_samples[:, cnt] = hddm_trace[param]
        cnt += 1
    
    return posterior_samples

def _make_trace_plotready_hierarchical(hddm_trace = None, model = ''):
    """Internal function to turn trace into data-format expected by plots. This
       version handles a hierarchical dataset with multiple subjects.

    Arguments:
        hddm_trace: panda DataFrame
            Output of the hddm.get_traces() function.
        model: str <default=''>
            Name of the model that was fit (e.g. 'ddm', 'weibull', 'levy' ...)
    Return:
        panda DataFrame: Adjusted traces.
    """
    
    subj_l = []
    for key in hddm_trace.keys():
        if '_subj' in key:
            new_key = pad_subj_id(key)
            #print(new_key)
            #new_key = key
            subj_l.append(str_to_num(new_key[-3:]))
            #subj_l.append(int(float(key[-3:])))

    dat = np.zeros((max((subj_l)) + 1, hddm_trace.shape[0], len(model_config[model]['params'])))
    for key in hddm_trace.keys():
        if '_subj' in key:
            new_key = pad_subj_id(key)
            
            id_tmp = str_to_num(new_key[-3:]) #int(float(key[-3:])) # convert padded key from string to a number
            if '_trans' in key:
                # isolate parameter name
                key_param_only = key.split('_')[0]
                lower_lim = model_config[model]['param_bounds'][0][model_config[model]['params'].index(key_param_only)]
                upper_lim = model_config[model]['param_bounds'][1][model_config[model]['params'].index(key_param_only)]
                val_tmp = lower_lim + (upper_lim - lower_lim) * (1 / ( 1 + np.exp(- hddm_trace[key])))
            else:
                val_tmp = hddm_trace[key]
            
            dat[id_tmp, : , model_config[model]['params'].index(key[:key.find('_')])] = val_tmp     
    return dat 

def _make_trace_plotready_condition(hddm_trace = None, model = ''):
    """Internal function to turn trace into data-format expected by plots. This
       version handles a dataset with multiple conditions.

    Arguments:
        hddm_trace: panda DataFrame
            Output of the hddm.get_traces() function.
        model: str <default=''>
            Name of the model that was fit (e.g. 'ddm', 'weibull', 'levy' ...)
    Return:
        panda DataFrame: Adjusted traces.
    """
    
    cond_l = []
    for key in hddm_trace.keys():
        if '(' in key:
            cond_l.append(int(float(key[-2])))
    
    dat = np.zeros((max(cond_l) + 1, hddm_trace.shape[0], len(model_config[model]['params'])))
                   
    for key in hddm_trace.keys():
        if '(' in key:
            id_tmp = int(float(key[-2]))
            if '_trans' in key:
                key_param_only = key.split('_')[0]
                lower_lim = model_config[model]['param_bounds'][0][model_config[model]['params'].index(key_param_only)]
                upper_lim = model_config[model]['param_bounds'][1][model_config[model]['params'].index(key_param_only)]
                val_tmp = lower_lim + (upper_lim - lower_lim) * (1 / ( 1 + np.exp(- hddm_trace[key])))
                #val_tmp = 1 / ( 1 + np.exp(- hddm_trace[key]))
                dat[id_tmp, : , model_config[model]['params'].index(key[:key.find('_trans')])] = val_tmp
            else:
                val_tmp = hddm_trace[key]
                dat[id_tmp, : , model_config[model]['params'].index(key[:key.find('(')])] = val_tmp   
        else:
            if '_trans' in key:
                key_param_only = key.split('_')[0]
                lower_lim = model_config[model]['param_bounds'][0][model_config[model]['params'].index(key_param_only)]
                upper_lim = model_config[model]['param_bounds'][1][model_config[model]['params'].index(key_param_only)]
                val_tmp = lower_lim + (upper_lim - lower_lim) * (1 / ( 1 + np.exp(- hddm_trace[key])))
                #val_tmp = 1 / ( 1 + np.exp(- hddm_trace[key]))
                key = key[:key.find('_trans')]
            else:
                val_tmp = hddm_trace[key]
                   
            dat[:, :, model_config[model]['params'].index(key)] = val_tmp
            
    return dat
# --------------------------------------------------------------------------------------------

# Plot bound
# Mean posterior predictives
def model_plot(posterior_samples = None,
               ground_truth_parameters = None,
               ground_truth_data = None,
               model_ground_truth = 'weibull_cdf',
               model_fitted = 'angle',
               input_is_hddm_trace = True,
               datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition' # data structure
               condition_column = 'condition', # data structure
               n_plots = 4, 
               n_posterior_parameters = 500,
               n_simulations_per_parameter = 10,
               cols = 3, # styling
               max_t = 5, # styling
               show_model = True, # styling
               show_trajectories = False, # styling
               n_trajectories = 10,
               color_trajectories = 'blue',
               alpha_trajectories = 0.2,
               linewidth_trajectories = 1.0,
               ylimit = 2, # styling
               posterior_linewidth = 3, # styling
               ground_truth_linewidth = 3, # styling
               hist_linewidth = 3, # styling
               bin_size = 0.025, # styling
               save = False,
               scale_x = 1.0,
               scale_y = 1.0,
               delta_t_graph = 0.01):
    
    """The model plot is useful to illustrate model behavior graphically. It is quite a flexible 
       plot allowing you to show path trajectories and embedded reaction time histograms etc.. 
       The main feature is the graphical illustration of a given model 
       (this works for 'ddm', 'ornstein', 'levy', 'weibull', 'angle') separately colored for the ground truth parameterization
       and the parameterizations supplied as posterior samples from a hddm sampling run. 

    Arguments:
        posterior_samples: panda.DataFrame <default=None>
            Holds the posterior samples. This will usually be the output of 
            hddm_model.get_traces().
        ground_truth_parameters: np.array <default=None>
            Array holding ground truth parameters. Depending on the structure supplied under the 
            datatype argument, this may be a 1d or 2d array.
        ground_truth_data: panda.DataFrame
            Ground truth dataset as supplied to hddm. Has a 'rt' column, a 'response' column and a 'subj_idx' column
            and potentially a 'condition' column.
        model_ground_truth: str <default='weibull_cdf'>
            String that speficies which model was the ground truth. This is useful mainly for parameter recovery excercises,
            one obviously doesn't usually have access to the ground truth.
        model_fitted: str <default='angle'>
            String that specifies which model the data was fitted to. This is necessary, for the plot to interpret the 
            supplied traces correctly, and to choose the correct simulator for visualization.
        datatype: str <default='single_subject'>
            Three options as of now. 'single_subject', 'hierarchical', 'condition'
        condition_column: str <default='condition'>
            The column that specifies the condition in the data supplied under the ground_truth_data argument.
        n_plots: int <default=4>
            The plot attempts to be smart in trying to figure out how many plots are desired, however choosing it manual is a 
            save option.
        n_posterior_parameters: int <default=500>
            Number of posterior samples to draw for plotting. This needs to be smaller or equal to the number 
            of posterior samples supplied to the plot.
        n_simulations_per_parameter: int <default=10>
            How many simulations to perform for each posterior parameter vector drawn.
        cols: int <default=3>
            Number of columns to split the plot into.
        max_t: float <default=10>
            Maximim reaction time to allow for internal simulations for the plot.
        show_model: bool <default=True>
            Whether or not to show the model in the final output (other option is to just show reaction time histograms)
        show_trajectories: bool <default=False>
            Whether or not to show some example trajectories of the simulators.
        n_trajectories: int <default=10>
            Number of trajectories to show if the show_trajectories argument is set to True,
        color_trajectories: str <default='blue'>
            Color of the trajectories if the show_trajectories arguent is set to True.
        alpha_trajectories: float <default=0.2>
            Sets transparency level of trajectories if the show_trajectories argument is set to True.
        linewidth_trajectories: float <default=1.0>
            Sets the linewidth of trajectories if the show_trajectories argument is set to True.
        ylimit: float <default=2>
            Sets the limit on the y-axis
        posterior_linewidth: float <default=3>
            Linewidth of the model visualizations corresponding to posterior samples.
        ground_truth_linewidth: float <default=3>
            Linewidth of the model visualization corresponding to the ground truth model
        hist_linewidth: float <default=3>
            Linewidth of the reaction time histograms (for gorund truth and posterior samples).
            To hide the reaction time histograms, set it to 0.
        bin_size: float <default=0.025>
            Bin size for the reaction time histograms.
        save: bool <default=False>
            Whether to save the plot
        scale_x: float <default=1.0>
            Scales the x axis of the graph
        scale_y: float <default=1.0>
            Salces the y axes o the graph
        delta_t_graph: float <default=0.01>
            Timesteps to use for the simulation runs performed for plotting.
        input_is_hddm_trace: bin <default=True>>
            Whether or not the posterior samples supplied are coming from hddm traces. 
            NOTE, this does not accept False as of now.
    Return: plot object
    """
    
    if save == True:
        pass
        # matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['svg.fonttype'] = 'none'

    # In case we don't fit 'z' we set it to 0.5 here for purposes of plotting 
    if posterior_samples is not None:
        z_cnt  = 0
        for ps_idx in posterior_samples.keys():
            if 'z' in ps_idx:
                z_cnt += 1
        if z_cnt < 1:
            posterior_samples['z_trans'] = 0.0
            print('z not part of fitted parameters --> Figures assume it was set to 0.5')
            
    # Inputs are hddm_traces --> make plot ready
    if input_is_hddm_trace and posterior_samples is not None:
        if datatype == 'single_subject':
            posterior_samples = _make_trace_plotready_single_subject(posterior_samples, 
                                                                     model = model_fitted)
        
        if datatype == 'hierarchical':
            posterior_samples = _make_trace_plotready_hierarchical(posterior_samples, 
                                                                   model = model_fitted)
            #print(posterior_samples.shape)
            n_plots = posterior_samples.shape[0]
#             print(posterior_samples)
        
        if datatype == 'condition':
            posterior_samples = _make_trace_plotready_condition(posterior_samples, 
                                                                model = model_fitted)
            n_plots = posterior_samples.shape[0]

    if posterior_samples is None and model_ground_truth is None:
        return 'Please provide either posterior samples, \n or a ground truth model and parameter set to plot something here. \n Currently you are requesting an empty plot' 
    
    # Taking care of special case with 1 plot
    if n_plots == 1:
        if model_ground_truth is not None:
            ground_truth_parameters = np.expand_dims(ground_truth_parameters, 0)
        if posterior_samples is not None:
            posterior_samples = np.expand_dims(posterior_samples, 0)
        if ground_truth_data is not None:
            gt_dat_dict = dict()
            gt_dat_dict[0] = ground_truth_data.values
            ground_truth_data = gt_dat_dict
            #ground_truth_data = np.expand_dims(ground_truth_data, 0)
            sorted_keys = [0]
            
    title = 'Model Plot: '
    
    if model_ground_truth is not None:
        ax_titles = model_config[model_ground_truth]['params']
    else: 
        ax_titles = ''
        
    if ground_truth_data is not None and datatype == 'condition':
        if condition_column is None:
            return 'Need to specify the name of the condition column'
        ####
        gt_dat_dict = dict()
        for i in np.sort(np.unique(ground_truth_data[condition_column])):
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data[condition_column] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict

        sorted_keys = np.sort(np.unique(ground_truth_data[condition_column]))
        ground_truth_data = gt_dat_dict

        print(sorted_keys)
        print(ground_truth_data.keys())
        # print('Supplying ground truth data not yet implemented for hierarchical datasets')
        
    
    # AF TODO: Generalize to arbitrary response coding !
    elif ground_truth_data is not None and datatype == 'hierarchical':
        gt_dat_dict = dict()
        
        for i in np.sort(np.unique(ground_truth_data['subj_idx'])):
            print(i)
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data['subj_idx'] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        
        sorted_keys = np.sort(np.unique(ground_truth_data['subj_idx']))
        ground_truth_data = gt_dat_dict
        
        print(sorted_keys)
        print(ground_truth_data.keys())
        # print('Supplying ground truth data not yet implemented for hierarchical datasets')

    # elif ground_truth_data is not None and datatype == 'single_subject':
    #     gt_dat_dict = dict()

    #     for 

    #     sorted_keys = np.sort(np.unique(ground_truth_data['subj_idx']()))


    # Define number of rows we need for display
    if n_plots > 1:
        rows = int(np.ceil(n_plots / cols))
    else:
        rows = 1

    # Some style settings
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20 * scale_x, 20 * rows * scale_y), 
                           sharex = False, 
                           sharey = False)
    
    # Title adjustments depending on whether ground truth model was supplied
    if model_ground_truth is not None:  
        my_suptitle = fig.suptitle(title + model_ground_truth, fontsize = 40)
    else:
        my_suptitle = fig.suptitle(title.replace(':', ''), fontsize = 40)
        
    sns.despine(right = True)

    t_s = np.arange(0, max_t, delta_t_graph)
    nbins = int((max_t) / bin_size)

    for i in range(n_plots):
        
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        if rows > 1 and cols > 1:
            ax[row_tmp, col_tmp].set_xlim(0, max_t)
            ax[row_tmp, col_tmp].set_ylim(- ylimit, ylimit)
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax[i].set_xlim(0, max_t)
            ax[i].set_ylim(-ylimit, ylimit)
        else:
            ax.set_xlim(0, max_t)
            ax.set_ylim(-ylimit, ylimit)

        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp]
            ax_tmp_twin_up = ax[row_tmp, col_tmp].twinx()
            ax_tmp_twin_down = ax[row_tmp, col_tmp].twinx()
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i]
            ax_tmp_twin_up = ax[i].twinx()
            ax_tmp_twin_down = ax[i].twinx()
        else:
            ax_tmp = ax
            ax_tmp_twin_up = ax.twinx()
            ax_tmp_twin_down = ax.twinx()
        
        ax_tmp_twin_up.set_ylim(-ylimit, ylimit)
        ax_tmp_twin_up.set_yticks([])

        ax_tmp_twin_down.set_ylim(ylimit, -ylimit)
        ax_tmp_twin_down.set_yticks([])
            
        # ADD TRAJECTORIESS
        if show_trajectories == True:
            for k in range(n_trajectories):
                out = simulator(theta = ground_truth_parameters[i, :],
                                model = model_ground_truth, 
                                n_samples = 1,
                                bin_dim = None)
                ax_tmp.plot(out[2]['ndt'] + np.arange(0, out[2]['max_t'] +  out[2]['delta_t'], out[2]['delta_t'])[out[2]['trajectory'][:, 0] > -999], 
                            out[2]['trajectory'][out[2]['trajectory'] > -999], 
                            color = color_trajectories, 
                            alpha = alpha_trajectories,
                            linewidth = linewidth_trajectories)

                #ax_ins = ax.inset_axes([1, 0.5, 0.2, 0.2]) --> important for levy ! AF TODO
                #ax_ins.plot([0, 1, 2, 3])
 
        # ADD HISTOGRAMS

        # RUN SIMULATIONS: GROUND TRUTH PARAMETERS
        if model_ground_truth is not None and ground_truth_data is None: # If ground truth model is supplied but not corresponding dataset --> we simulate one
            out = simulator(theta = ground_truth_parameters[i, :],
                            model = model_ground_truth, 
                            n_samples = 20000,
                            bin_dim = None)
             
            tmp_true = np.concatenate([out[0], out[1]], axis = 1)
            choice_p_up_true = np.sum(tmp_true[:, 1] == 1) / tmp_true.shape[0]
        
        # RUN SIMULATIONS: POSTERIOR SAMPLES
        if posterior_samples is not None:
            
            # Run Model simulations for posterior samples
            tmp_post = np.zeros((n_posterior_parameters * n_simulations_per_parameter, 2))
            idx = np.random.choice(posterior_samples.shape[1], size = n_posterior_parameters, replace = False)

            for j in range(n_posterior_parameters):
                out = simulator(theta = posterior_samples[i, idx[j], :],
                                model = model_fitted,
                                n_samples = n_simulations_per_parameter,
                                bin_dim = None)
                                
                tmp_post[(n_simulations_per_parameter * j):(n_simulations_per_parameter * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)

        # DRAW DATA HISTOGRAMS
        if posterior_samples is not None:
            choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]

            counts_2_up, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)

            counts_2_down, _ = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)
            
            if j == (n_posterior_parameters - 1) and row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Posterior Predictive'
            else:
                tmp_label = None

            ax_tmp_twin_up.hist(bins[:-1], 
                                bins, 
                                weights = choice_p_up_post * counts_2_up,
                                histtype = 'step',
                                alpha = 0.5, 
                                color = 'black',
                                edgecolor = 'black',
                                zorder = -1,
                                label = tmp_label,
                                linewidth = hist_linewidth)

            ax_tmp_twin_down.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_post) * counts_2_down,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'black',
                        edgecolor = 'black',
                        linewidth = hist_linewidth,
                        zorder = -1)
            
                       
        if model_ground_truth is not None and ground_truth_data is None:
            counts_2_up, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                             bins = np.linspace(0, max_t, nbins),
                                             density = True)

            counts_2_down, _ = np.histogram(tmp_true[tmp_true[:, 1] == - 1, 0],
                                            bins = np.linspace(0, max_t, nbins),
                                            density = True)

            if row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Ground Truth Data'
            else: 
                tmp_label = None
            
            ax_tmp_twin_up.hist(bins[:-1], 
                                bins, 
                                weights = choice_p_up_true * counts_2_up,
                                histtype = 'step',
                                alpha = 0.5, 
                                color = 'red',
                                edgecolor = 'red',
                                zorder = -1,
                                linewidth = hist_linewidth,
                                label = tmp_label)

            ax_tmp_twin_down.hist(bins[:-1], 
                                  bins, 
                                  weights = (1 - choice_p_up_true) * counts_2_down,
                                  histtype = 'step',
                                  alpha = 0.5, 
                                  color = 'red',
                                  edgecolor = 'red',
                                  linewidth = hist_linewidth,
                                  zorder = -1)
 
            if row_tmp == 0 and col_tmp == 0:
                ax_tmp_twin_up.legend(loc = 'lower right')
            
        if ground_truth_data is not None:
            # These splits here is neither elegant nor necessary --> can represent ground_truth_data simply as a dict !
            # Wiser because either way we can have varying numbers of trials for each subject !
            print('sorted keys')
            print(sorted_keys)

            print('ground truth data')
            print(ground_truth_data)
            print(ground_truth_data[sorted_keys[i]])
            print(type(ground_truth_data[sorted_keys[i]]))
            print(ground_truth_data[sorted_keys[i]][:, 1] == 1)
            print(ground_truth_data[sorted_keys[i]][ground_truth_data[sorted_keys[i]][:, 1] == 1, 0])
            counts_2_up, bins = np.histogram(ground_truth_data[sorted_keys[i]][ground_truth_data[sorted_keys[i]][:, 1] == 1, 0],
                                            bins = np.linspace(0, max_t, nbins),
                                            density = True)

            counts_2_down, _ = np.histogram(ground_truth_data[sorted_keys[i]][ground_truth_data[sorted_keys[i]][:, 1] == - 1, 0],
                                            bins = np.linspace(0, max_t, nbins),
                                            density = True)

            choice_p_up_true_dat = np.sum(ground_truth_data[sorted_keys[i]][:, 1] == 1) / ground_truth_data[sorted_keys[i]].shape[0]

            if row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Dataset'
            else:
                tmp_label = None
            
            ax_tmp_twin_up.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_true_dat * counts_2_up,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'blue',
                            edgecolor = 'blue',
                            zorder = -1,
                            linewidth = hist_linewidth,
                            label = tmp_label)

            ax_tmp_twin_down.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_true_dat) * counts_2_down,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'blue',
                        edgecolor = 'blue',
                        linewidth = hist_linewidth,
                        zorder = -1)
            
            if row_tmp == 0 and col_tmp == 0:
                ax_tmp_twin_up.legend(loc = 'lower right')

        # POSTERIOR SAMPLES: BOUNDS AND SLOPES (model)
        if show_model:
            if posterior_samples is None:
                # If we didn't supply posterior_samples but want to show model
                # we set n_posterior_parameters to 1 and should be 
                n_posterior_parameters = 0
                for j in range(n_posterior_parameters + 1):
                    tmp_label = ""
                    if j == (n_posterior_parameters - 1):
                        tmp_label = 'Model Samples'
                        tmp_model = model_fitted
                        tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.5
                        tmp_color = 'black'
                        tmp_linewidth = posterior_linewidth
                    elif j == n_posterior_parameters and model_ground_truth is not None:
                        tmp_samples = ground_truth_parameters[i, :]
                        tmp_model = model_ground_truth
                        
                        # If we supplied ground truth data --> make ground truth model blue, otherwise red
                        tmp_colors = ['red', 'blue']
                        tmp_bool = ground_truth_data is not None
                        tmp_color = tmp_colors[int(tmp_bool)]
                        tmp_alpha = 1
                        tmp_label = 'Ground Truth Model'
                        tmp_linewidth = ground_truth_linewidth
                    elif j == n_posterior_parameters and model_ground_truth == None:
                        break
                    else:
                        tmp_model = model_fitted
                        tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.05
                        tmp_color = 'black'
                        tmp_label = None
                        tmp_linewidth = posterior_linewidth

                    print(tmp_label)
                    
                    # MAKE BOUNDS (FROM MODEL CONFIG) !
                    if tmp_model == 'weibull_cdf' or tmp_model == 'weibull_cdf2' or tmp_model == 'weibull_cdf_concave' or tmp_model == 'weibull':
                        b = np.maximum(tmp_samples[1] * model_config[tmp_model]['boundary'](t = t_s, 
                                                                                            alpha = tmp_samples[4],
                                                                                            beta = tmp_samples[5]), 0)

                    if tmp_model == 'angle' or tmp_model == 'angle2':
                        b = np.maximum(tmp_samples[1] + model_config[tmp_model]['boundary'](t = t_s, theta = tmp_samples[4]), 0)
                    
                    if tmp_model == 'ddm' or tmp_model == 'ornstein' or tmp_model == 'levy' or tmp_model == 'full_ddm':
                        b = tmp_samples[1] * np.ones(t_s.shape[0]) #model_config[tmp_model]['boundary'](t = t_s)                   


                    # MAKESLOPES (VIA TRAJECTORIES) !
                    out = simulator(theta = tmp_samples,
                                    model = tmp_model, 
                                    n_samples = 1,
                                    no_noise = True,
                                    delta_t = delta_t_graph,
                                    bin_dim = None)
                    
                    tmp_traj = out[2]['trajectory']
                    maxid = np.minimum(np.argmax(np.where(tmp_traj > - 999)), t_s.shape[0])

                    ax_tmp.plot(t_s + tmp_samples[model_config[tmp_model]['params'].index('t')], b, tmp_color,
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = tmp_linewidth,
                                label = tmp_label,
                                )

                    ax_tmp.plot(t_s + tmp_samples[model_config[tmp_model]['params'].index('t')], -b, tmp_color, 
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = tmp_linewidth,
                                )

                    ax_tmp.plot(t_s[:maxid] + tmp_samples[model_config[tmp_model]['params'].index('t')],
                                tmp_traj[:maxid],
                                c = tmp_color, 
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = tmp_linewidth) # TOOK AWAY LABEL

                    ax_tmp.axvline(x = tmp_samples[model_config[tmp_model]['params'].index('t')], # this should identify the index of ndt directly via model config !
                                   ymin = - ylimit, 
                                   ymax = ylimit, 
                                   c = tmp_color, 
                                   linestyle = '--',
                                   linewidth = tmp_linewidth,
                                   alpha = tmp_alpha)

                    if tmp_label == 'Ground Truth Model' and row_tmp == 0 and col_tmp == 0:
                        ax_tmp.legend(loc = 'upper right')
                        print('generated upper right label')
                        print('row: ', row_tmp)
                        print('col: ', col_tmp)
                        print('j: ', j)

                    if rows == 1 and cols == 1:
                        ax_tmp.patch.set_visible(False)
                    
        # Set plot title
        title_tmp = ''
        if n_plots > 1:
            title_tmp += 'S ' + str(i) + ' '
        if model_ground_truth is not None:
            for k in range(len(ax_titles)):
                title_tmp += ax_titles[k] + ': '
                if k == (len(ax_titles)  - 1):
                    title_tmp += str(round(ground_truth_parameters[i, k], 2))
                else:
                    title_tmp += str(round(ground_truth_parameters[i, k], 2)) + ', '

        if row_tmp == (rows - 1):
            ax_tmp.set_xlabel('rt', 
                              fontsize = 20);
        ax_tmp.set_ylabel('', 
                          fontsize = 20);

        ax_tmp.set_title(title_tmp,
                         fontsize = 24)
        ax_tmp.tick_params(axis = 'y', size = 20)
        ax_tmp.tick_params(axis = 'x', size = 20)

        # Some extra styling:
        if model_ground_truth is not None:
            if show_model:
                ax_tmp.axvline(x = ground_truth_parameters[i, model_config[model_ground_truth]['params'].index('t')], ymin = - ylimit, ymax = ylimit, c = 'red', linestyle = '--')
            ax_tmp.axhline(y = 0, xmin = 0, xmax = ground_truth_parameters[i, model_config[model_ground_truth]['params'].index('t')] / max_t, c = 'red',  linestyle = '--')
        
    if rows > 1 and cols > 1:
        for i in range(n_plots, rows * cols, 1):
            row_tmp = int(np.floor(i / cols))
            col_tmp = i - (cols * row_tmp)
            ax[row_tmp, col_tmp].axis('off')

    plt.tight_layout(rect = [0, 0.03, 1, 0.9])
    
    if save == True:
        plt.savefig('figures/' + 'hierarchical_model_plot_' + model_ground_truth + '_' + datatype + '.png',
                    format = 'png', 
                    transparent = True,
                    frameon = False)
        plt.close()
    
    return plt.show()

def posterior_predictive_plot(posterior_samples = None,
                              ground_truth_parameters = None,
                              ground_truth_data = None,
                              n_plots = 9,
                              cols = 3,
                              model_fitted = 'angle',
                              model_ground_truth = 'angle',
                              datatype = 'single_subject',
                              condition_column = 'condition',
                              input_is_hddm_trace = True,
                              n_posterior_parameters = 100,
                              max_t = 20,
                              n_simulations_per_parameter = 10,
                              xlimit = 10,
                              bin_size = 0.025,
                              hist_linewidth = 3,
                              scale_x = 0.5,
                              scale_y = 0.5,
                              save = False):
    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        posterior_samples: panda.DataFrame <default=None>
            Holds the posterior samples. This will usually be the output of 
            hddm_model.get_traces().
        ground_truth_parameters: np.array <default=None>
            Array holding ground truth parameters. Depending on the structure supplied under the 
            datatype argument, this may be a 1d or 2d array.
        ground_truth_data: panda.DataFrame
            Ground truth dataset as supplied to hddm. Has a 'rt' column, a 'response' column and a 'subj_idx' column
            and potentially a 'condition' column.
        model_ground_truth: str <default='weibull_cdf'>
            String that speficies which model was the ground truth. This is useful mainly for parameter recovery excercises,
            one obviously doesn't usually have access to the ground truth.
        model_fitted: str <default='angle'>
            String that specifies which model the data was fitted to. This is necessary, for the plot to interpret the 
            supplied traces correctly, and to choose the correct simulator for visualization.
        datatype: str <default='single_subject'>
            Three options as of now. 'single_subject', 'hierarchical', 'condition'
        condition_column: str <default='condition'>
            The column that specifies the condition in the data supplied under the ground_truth_data argument.
        n_plots: int <default=4>
            The plot attempts to be smart in trying to figure out how many plots are desired, however choosing it manual is a 
            save option.
        n_posterior_parameters: int <default=500>
            Number of posterior samples to draw for plotting. This needs to be smaller or equal to the number 
            of posterior samples supplied to the plot.
        n_simulations_per_parameter: int <default=10>
            How many simulations to perform for each posterior parameter vector drawn.
        cols: int <default=3>
            Number of columns to split the plot into.
        max_t: float <default=10>
            Maximim reaction time to allow for internal simulations for the plot.
        xlimit: float <default=2>
            Sets the limit on the x-axis
        hist_linewidth: float <default=3>
            Linewidth of the reaction time histograms (for gorund truth and posterior samples).
            To hide the reaction time histograms, set it to 0.
        bin_size: float <default=0.025>
            Bin size for the reaction time histograms.
        save: bool <default=False>
            Whether to save the plot
        scale_x: float <default=1.0>
            Scales the x axis of the graph
        scale_y: float <default=1.0>
            Salces the y axes o the graph
        delta_t_graph: float <default=0.01>
            Timesteps to use for the simulation runs performed for plotting.
        input_is_hddm_trace: bin <default=True>>
            Whether or not the posterior samples supplied are coming from hddm traces. 
            NOTE, this does not accept False as of now.
    Return: plot object
    """
    
    if save == True:
        pass
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['pdf.fonttype'] = 42
        #matplotlib.rcParams['svg.fonttype'] = 'none'
    
    if model_ground_truth is None and ground_truth_data is None and posterior_samples is None:
        return 'No ground truth model was supplied, no dataset was supplied and no posterior sample was supplied. Nothing to plot' 

    # Inputs are hddm_traces --> make plot ready
    if input_is_hddm_trace and posterior_samples is not None:
        if datatype == 'hierarchical':
            posterior_samples = _make_trace_plotready_hierarchical(posterior_samples, 
                                                                   model = model_fitted)
            n_plots = posterior_samples.shape[0]
#             print(posterior_samples)
            
        if datatype == 'single_subject':
            posterior_samples = _make_trace_plotready_single_subject(posterior_samples, 
                                                                     model = model_fitted)
        if datatype == 'condition':
            posterior_samples = _make_trace_plotready_condition(posterior_samples, 
                                                                model = model_fitted)
            n_plots = posterior_samples.shape[0]

    # Taking care of special case with 1 plot
    if n_plots == 1:
        if model_ground_truth is not None:
            ground_truth_parameters = np.expand_dims(ground_truth_parameters, 0)
        if posterior_samples is not None:
            posterior_samples = np.expand_dims(posterior_samples, 0) # Seems unnecessary
        if ground_truth_data is not None:
            label_idx = [0] #np.unique(ground_truth_data['subj_idx'])
            gt_dat_dict = dict()
            gt_dat_dict[0] = ground_truth_data
            ground_truth_data = gt_dat_dict
            
            #ground_truth_data = np.expand_dims(ground_truth_data, 0)     
    
    # Take care of ground_truth_data
    #label_idx = list()
    if ground_truth_data is not None and datatype == 'hierarchical':
        # initialize ground truth data dictionary
        gt_dat_dict = dict()
        
        # Collect and keep relevant labels for later use
        label_idx = list()
        
        for i in np.sort(np.unique(ground_truth_data['subj_idx'])):
            label_idx.append(i)
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data['subj_idx'] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict
     
    if ground_truth_data is not None and datatype == 'condition':
        # initialize ground truth data dictionary
        gt_dat_dict = dict()
        
        # Collect and keep relevant labels for later use
        label_idx = list()

        for i in np.sort(np.unique(ground_truth_data[condition_column])):
            label_idx.append(i)
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data[condition_column] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict

    # Taking care of special case with 1 plot
    if n_plots == 1:
        cols = 1
    
    # if n_plots == 1:
    #     if model_ground_truth is not None:
    #         ground_truth_parameters = np.expand_dims(ground_truth_parameters, 0)
    #     if posterior_samples is not None:
    #         posterior_samples = np.expand_dims(posterior_samples, 0)
    #     if ground_truth_data is not None:
    #         ground_truth_data = np.expand_dims(ground_truth_data, 0)

    print(posterior_samples.shape)
 
    # General plot parameters
    nbins = int((2 * max_t) / bin_size)     
    rows = int(np.ceil(n_plots / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20 * scale_x, 20 * rows * scale_y), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Posterior Predictive: ' + model_fitted.upper(),
                 fontsize = 24)
    
    sns.despine(right = True)

    # Cycle through plots
    for i in range(n_plots):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        post_tmp = np.zeros((n_posterior_parameters * n_simulations_per_parameter, 2))
        idx = np.random.choice(posterior_samples.shape[1], 
                               size = n_posterior_parameters, 
                               replace = False)

        # Run Model simulations for posterior samples
        print('Simulations for plot: ', i)
        for j in range(n_posterior_parameters):
            out = simulator(theta = posterior_samples[i, idx[j], :], 
                            model = model_fitted,
                            n_samples = n_simulations_per_parameter,
                            bin_dim = None)
          
            post_tmp[(n_simulations_per_parameter * j):(n_simulations_per_parameter * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
        
        # Run Model simulations for true parameters
        # Supply data too !
        if model_ground_truth is not None and ground_truth_data is None:
            out = simulator(theta = ground_truth_parameters[i, :],
                            model = model_ground_truth,
                            n_samples = 20000,
                            bin_dim = None)
  
            gt_tmp = np.concatenate([out[0], out[1]], axis = 1)
            gt_color = 'red'
            #print('passed through')
        elif ground_truth_data is not None:
            gt_tmp = ground_truth_data[label_idx[i]].values # using the relevant label here instead of the plot number 
            gt_tmp[:, 1][gt_tmp[:,1] == 0.0] = -1.0 # set zero choices to -1 
            print(gt_tmp)
            gt_color = 'blue'

        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp]
        
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i]
        
        else:
            ax_tmp = ax
        
        # ACTUAL PLOTTING
        ax_tmp.hist(post_tmp[:, 0] * post_tmp[:, 1], 
                    bins = np.linspace(- max_t, max_t, nbins), #50, # kde = False, # rug = False, 
                    alpha =  1, 
                    color = 'black',
                    histtype = 'step', 
                    density = 1, 
                    edgecolor = 'black',
                    linewidth = hist_linewidth
                    )
        if ground_truth_data is not None:
            ax_tmp.hist(gt_tmp[:, 0] * gt_tmp[:, 1], 
                        alpha = 0.5, 
                        color = gt_color, 
                        density = 1, 
                        edgecolor = gt_color,  
                        histtype = 'step',
                        linewidth = hist_linewidth, 
                        bins = np.linspace(-max_t, max_t, nbins), #50, 
                        # kde = False, #rug = False,
                        )
        
        # EXTRA STYLING    
        ax_tmp.set_xlim(- xlimit, xlimit)
            
        if row_tmp == 0 and col_tmp == 0:
            if model_ground_truth is not None:
                label_0 = 'Ground Truth'
            else:
                label_0 = 'DATA'
            ax_tmp.legend(labels = ['Posterior Predictive', label_0], 
                          fontsize = 12, 
                          loc = 'upper right')
            
        if row_tmp == (rows - 1):
            ax_tmp.set_xlabel('rt', 
                              fontsize = 24)

        if col_tmp == 0:
            ax_tmp.set_ylabel('', 
                              fontsize = 24)

        ax_tmp.tick_params(axis = 'y', size = 22)
        ax_tmp.tick_params(axis = 'x', size = 22)
        
    if rows > 1 and cols > 1:
        for i in range(n_plots, rows * cols, 1):
            row_tmp = int(np.floor(i / cols))
            col_tmp = i - (cols * row_tmp)
            ax[row_tmp, col_tmp].axis('off')    
            
    if save == True:
        plt.savefig('figures/' + 'posterior_predictive_plot_' + model_ground_truth + '_' + datatype + '.png',
                    format = 'png', 
                    transparent = True,
                    frameon = False)
        plt.close()

    return plt.show()

def caterpillar_plot(posterior_samples = [],
                     ground_truth_parameters = None,
                     model_fitted = 'angle',
                     datatype = 'hierarchical', # 'hierarchical', 'single_subject', 'condition'
                     drop_sd = True,
                     keep_key = None,
                     x_limits = [-2, 2],
                     aspect_ratio = 2,
                     figure_scale = 1.0,
                     save = False,
                     tick_label_size_x = 22,
                     tick_label_size_y = 14):

    """An alternative posterior predictive plot. Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        posterior_samples: panda.DataFrame <default=None>
            Holds the posterior samples. This will usually be the output of 
            hddm_model.get_traces().
        ground_truth_parameters: np.array <default=None>
            Array holding ground truth parameters. Depending on the structure supplied under the 
            datatype argument, this may be a 1d or 2d array.
        model_fitted: str <default='angle'>
            String that specifies which model the data was fitted to. This is necessary, for the plot to interpret the 
            supplied traces correctly, and to choose the correct simulator for visualization.
        datatype: str <default='single_subject'>
            Three options as of now. 'single_subject', 'hierarchical', 'condition'
        drop_sd: bool <default=True>
            Whether or not to drop group level standard deviations from the caterpillar plot.
            This is sometimes useful because scales can be off if included.
        keep_key: list <default=None>
            If you want to keep only a specific list of parameters in the caterpillar plot, supply those here as 
            a list. All other parameters for which you supply traces in the posterior samples are going to be ignored.
        x_limits: float <default=2>
            Sets the limit on the x-axis
        aspect_ratio: float <default=2>
            Aspect ratio of plot.
        figure_scale: float <default=1.0>
            Figure scaling. 1.0 refers to 10 inches as a baseline size.
        tick_label_size_x: int <default=22>
            Basic plot styling. Tick label size.
        tick_label_size_y: int <default=14>
            Basic plot styling. Tick label size.
        save: bool <default=False>
            Whether to save the plot

    Return: plot object
    """
    
    if save == True:
        pass
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['pdf.fonttype'] = 42
        #matplotlib.rcParams['svg.fonttype'] = 'none'
    
    sns.set(style = "white", 
        palette = "muted", 
        color_codes = True,
        font_scale = 2)
    
    fig, ax = plt.subplots(1, 1, 
                           figsize = (10 * figure_scale, aspect_ratio * 10 * figure_scale), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Caterpillar plot: ' + model_fitted.upper().replace('_', '-'), fontsize = 40)
    sns.despine(right = True)
    
    trace = posterior_samples.copy()
    
    # In case ground truth parameters were supplied --> this is mostly of interest for parameter recovery studies etc.
    if ground_truth_parameters is not None:
        cnt = 0
        gt_dict = {}
        
        if datatype == 'single_subject':
            if type(ground_truth_parameters) is not dict:
                for v in model_config[model_fitted]['params']:
                    gt_dict[v] = ground_truth_parameters[cnt]
                    cnt += 1
            else:
                gt_dict = ground_truth_parameters

        if datatype == 'hierarchical':
            gt_dict = ground_truth_parameters

        if datatype == 'condition':
            gt_dict = ground_truth_parameters
             
    ecdfs = {}
    plot_vals = {} # [0.01, 0.9], [0.01, 0.99], [mean]
    
    for k in trace.keys():
        # If we want to keep only a specific parameter we skip all traces which don't include it in 
        # their names !
        if keep_key is not None and keep_key not in k: 
            continue

        # Deal with 
        if 'std' in k and drop_sd:
            pass
        
        else:
            # Deal with _transformed parameters
            if '_trans' in k:
                label_tmp = k.replace('_trans', '')

                key_param_only = k.split('_')[0]
                print(key_param_only)
                print(k)
                lower_lim = model_config[model_fitted]['param_bounds'][0][model_config[model_fitted]['params'].index(key_param_only)]
                upper_lim = model_config[model_fitted]['param_bounds'][1][model_config[model_fitted]['params'].index(key_param_only)]
                trace[label_tmp] = lower_lim + (upper_lim - lower_lim) * (1 / ( 1 + np.exp(- trace[k])))

                #trace[label_tmp] = 1 / (1 + np.exp(- trace[k]))
                k = label_tmp

            ok_ = 1
            k_old = k # keep original key around for indexing
            k = k.replace('_', '-') # assign new prettier key for plotting
            
            if drop_sd == True:
                if 'sd' in k:
                    ok_ = 0
            if ok_:
                ecdfs[k] = ECDF(trace[k_old])
                tmp_sorted = sorted(trace[k_old])
                _p01 =  tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.01) - 1]
                _p99 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.99) - 1]
                _p1 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.1) - 1]
                _p9 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.9) - 1]
                _pmean = trace[k_old].mean()
                plot_vals[k] = [[_p01, _p99], [_p1, _p9], _pmean]
    
    x = [plot_vals[k][2] for k in plot_vals.keys()]
    ax.scatter(x, plot_vals.keys(), c = 'black', marker = 's', alpha = 0)
    
    for k in plot_vals.keys():
        k = k.replace('_', '-')
        ax.plot(plot_vals[k][1], [k, k], c = 'grey', zorder = - 1, linewidth = 5)
        ax.plot(plot_vals[k][0] , [k, k], c = 'black', zorder = - 1)
        
        if ground_truth_parameters is not None:
            ax.scatter(gt_dict[k.replace('-', '_')], k,  c = 'red', marker = "|")
        
    ax.set_xlim(x_limits[0], x_limits[1])
    ax.tick_params(axis = 'y', size = tick_label_size_y)
    ax.tick_params(axis = 'x', size = tick_label_size_x)
        
    if save == True:
        plt.savefig('figures/' + 'caterpillar_plot_' + model + '_' + datatype + '.png',
                    format = 'png', 
                    transparent = True,
                    frameon = False)

    return plt.show()

# Posterior Pair Plot
def posterior_pair_plot(posterior_samples = None, # Here expects single subject's posterior samples as panda dataframe (dictionary may work)
                        axes_limits = 'samples', # 'samples' or dict({'parameter_name': [lower bound, upper bound]})
                        height = 10,
                        aspect_ratio = 1,
                        n_subsample = 1000,
                        ground_truths = None,
                        model_fitted = None,
                        save = False):

    """Basic pair plot useful for inspecting posterior parameters.
       At this point can be used only for single subject data. 
       Works for all models listed in hddm (e.g. 'ddm', 'angle', 'weibull', 'levy', 'ornstein')

    Arguments:
        posterior_samples: pandas.DataFrame <default=None>
            Supplies the posterior samples as a pandas DataFrame with columns determining the parameters for which rows 
            store traces.
        axes_limits: str or dict <default='samples'>
            Either a string that says 'samples', which makes axes limits depends on the posterior sample values directly,
            (separately for each parameter). Or a dictionary with keys parameter names, and values a 2-list with a lower
            and an upper bound ([lower, upper]).
        height: float <default=10>
            Figure height in inches.
        aspect_ratio: float <default=1>
            Aspect ratio of figure 
        n_subsample: int <default=1000>
            Number of posterior samples to use for figure finally. Subsamples from the provided traces.
        ground_truths: list or numpy.array <default=None>
            Ground truth parameters (will be shown in the plot if supplied)
        model_fitted: str <default=None>
            String that supplies which model was fitted to the data.
        save: bool <default= False>
            Whether or not to save the figure.
    Return: plot object
    """
    
    if save == True:
        pass
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['pdf.fonttype'] = 42
        #matplotlib.rcParams['svg.fonttype'] = 'none'
    
    # some preprocessing
    #posterior_samples = posterior_samples.get_traces().copy()
    # Adjust this to be correct adjustment of z !!!! AF TODO
    for k in posterior_samples.keys():
        if '_trans' in k:
            label_tmp = k.replace('_trans', '')
            key_param_only = k.split('_')[0]
            lower_lim = model_config[model_fitted]['param_bounds'][0][model_config[model_fitted]['params'].index(key_param_only)]
            upper_lim = model_config[model_fitted]['param_bounds'][1][model_config[model_fitted]['params'].index(key_param_only)]
            posterior_samples[label_tmp] = lower_lim + (upper_lim - lower_lim) * (1 / ( 1 + np.exp(- posterior_samples[k])))
            posterior_samples = posterior_samples.drop(k, axis = 1)
            #trace[label_tmp] = 1 / (1 + np.exp(- trace[k]))

    #posterior_samples['z'] = 1 / ( 1 + np.exp(- posterior_samples['z_trans']))
    #posterior_samples = posterior_samples.drop('z_trans', axis = 1)

    g = sns.PairGrid(posterior_samples.sample(n_subsample), 
                     height = height / len(list(posterior_samples.keys())),
                     aspect = aspect_ratio,
                     diag_sharey = False)
    g = g.map_diag(sns.kdeplot, color = 'black', shade = False) # shade = True, 
    g = g.map_lower(sns.kdeplot, 
                    thresh = 0.01,
                    n_levels = 50,
                    shade = False,
                    cmap = 'Purples_d') # 'Greys'
    
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    # Get x and y labels of graph as determined by the posterior_samples panda
    xlabels, ylabels = [], []

    for ax in g.axes[-1, :]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)

    for ax in g.axes[:, 0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
    
    if axes_limits == 'model':
        for i in range(len(xlabels)):
            for j in range(len(ylabels)):
                try:
                    g.axes[j,i].set_xlim(model_config[model_fitted]['param_bounds'][0][model_config[model_fitted]['params'].index(xlabels[i])], 
                                        model_config[model_fitted]['param_bounds'][1][model_config[model_fitted]['params'].index(xlabels[i])])
                    g.axes[j,i].set_ylim(model_config[model_fitted]['param_bounds'][0][model_config[model_fitted]['params'].index(ylabels[j])], 
                                        model_config[model_fitted]['param_bounds'][1][model_config[model_fitted]['params'].index(ylabels[j])])
                except:
                    print('ERROR: It looks like you are trying to make axis limits dependend on model specific parameters, but the column-names of your posterior traces do not align with the requested model\'s parameters')
    
    elif type(axes_limits) == dict:
        for i in range(len(xlabels)):
            for j in range(len(ylabels)):
                try:
                    g.axes[j,i].set_xlim(axes_limits[xlabels[i]][0], 
                                         axes_limits[xlabels[i]][1])
                    g.axes[j,i].set_ylim(axes_limits[ylabels[j]][0], 
                                         axes_limits[ylabels[j]][1])
                except:
                    print('ERROR: Does your axes_limits dictionary match the column names of your posterior_samples DataFrame?')
                    return
    
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation = 45)

    g.fig.suptitle(model_fitted.upper(), 
                   y = 1.03, 
                   fontsize = 24)
    
    posterior_samples_key_set = np.sort(posterior_samples.keys())
    # If ground truth is available add it in:
    if ground_truths is not None:
        for i in range(g.axes.shape[0]):
            for j in range(i + 1, g.axes.shape[0], 1):
                g.axes[j,i].plot(ground_truths[xlabels[i]], 
                                 ground_truths[ylabels[j]], #posterior_samples_key_set[j]], 
                                 '.', 
                                 color = 'red',
                                 markersize = 10)
                # g.axes[j,i].plot(ground_truths[model_config[model_fitted]['params'].index(xlabels[i])], 
                #                  ground_truths[model_config[model_fitted]['params'].index(ylabels[j])], 
                #                  '.', 
                #                  color = 'red',
                #                  markersize = 10)

        for i in range(g.axes.shape[0]):
            # g.axes[i,i].plot(ground_truths[model_config[model_fitted]['params'].index(xlabels[i])],
            #                  g.axes[i,i].get_ylim()[0], 
            #                  '.', 
            #                  color = 'red',
            #                  markersize = 10)

            g.axes[i,i].plot(ground_truths[xlabels[i]],
                             g.axes[i,i].get_ylim()[0], 
                             '.', 
                             color = 'red',
                             markersize = 10)
            
    if save == True:
        plt.savefig('figures/' + 'pair_plot_' + model_fitted + '_' + datatype + '.png',
                    format = 'png', 
                    transparent = True,
                    frameon = False)
        plt.close()
            
    # Show
    return plt.show(block = False)

# STRUCTURE
# EXPECT A HDDM MODEL
# TRANSFORM TRACES INTO USABLE FORM
# PRODUCE PLOT