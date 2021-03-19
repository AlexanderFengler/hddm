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

def run_simulator():
    return simulator([0, 1, 0.5, 0.3, 0.3],
                            model = 'angle',
                            n_samples = 10000,
                            n_trials = 1,
                            delta_t = 0.001,
                            max_t = 20,
                            cartoon = False,
                            bin_dim = None, 
                            bin_pointwise = False)


# Plot preprocessing functions
def _make_trace_plotready_single_subject(hddm_trace = None, model = ''):
    
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
               input_is_hddm_trace = False,
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
               gt_linewidth = 3, # styling
               hist_linewidth = 3, # styling
               bin_size = 0.025, # styling
               save = False,
               scale_x = 1.0,
               scale_y = 1.0):
    
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
            
    # Inputs are hddm_traces --> make plot ready
    if input_is_hddm_trace and posterior_samples is not None:
        if datatype == 'hierarchical':
            posterior_samples = _make_trace_plotready_hierarchical(posterior_samples, 
                                                                   model = model_fitted)
            #print(posterior_samples.shape)
            n_plots = posterior_samples.shape[0]
#             print(posterior_samples)
            
        if datatype == 'single_subject':
            posterior_samples = _make_trace_plotready_single_subject(posterior_samples, 
                                                                     model = model_fitted)
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
            gt_dat_dict[0] = ground_truth_data
            ground_truth_data = gt_dat_dict
            #ground_truth_data = np.expand_dims(ground_truth_data, 0)
            
    
    title = 'Model Plot: '
    
    if model_ground_truth is not None:
        ax_titles = model_config[model_ground_truth]['params']
    else: 
        ax_titles = ''
        
    if ground_truth_data is not None and datatype == 'condition':
        ####
        gt_dat_dict = dict()
        for i in np.sort(np.unique(ground_truth_data[condition_column])):
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data[condition_column] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict
        
    
    # AF TODO: Generalize to arbitrary response coding !
    if ground_truth_data is not None and datatype == 'hierarchical':
        gt_dat_dict = dict()
        
        for i in np.sort(np.unique(ground_truth_data['subj_idx'])):
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data['subj_idx'] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        
        sorted_keys = np.sort(np.unique(ground_truth_data['subj_idx']))
        ground_truth_data = gt_dat_dict
        
        print(sorted_keys)
        print(ground_truth_data.keys())
        # print('Supplying ground truth data not yet implemented for hierarchical datasets')

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
                           figsize = (20 * rows * scale_y, 20 * scale_x), 
                           sharex = False, 
                           sharey = False)
    
    # Title adjustments depending on whether ground truth model was supplied
    if model_ground_truth is not None:  
        my_suptitle = fig.suptitle(title + model_ground_truth, fontsize = 40)
    else:
        my_suptitle = fig.suptitle(title.replace(':', ''), fontsize = 40)
        
    sns.despine(right = True)

    t_s = np.arange(0, max_t, 0.01)
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


        # Run simulations and add trajectories
        # if show_trajectories == True:
        #     for k in range(n_trajectories):
        #         out = simulator(theta = ground_truth_parameters[i, :],
        #                         model = model_ground_truth, 
        #                         n_samples = 1,
        #                         bin_dim = None)
        #         #print(out)
        #         if rows > 1 and cols > 1:
        #             ax[row_tmp, col_tmp].plot(out[2]['ndt'] + np.arange(0, out[2]['max_t'] +  out[2]['delta_t'], out[2]['delta_t'])[out[2]['trajectory'][:, 0] > -999], out[2]['trajectory'][out[2]['trajectory'] > -999], 
        #                                       color = color_trajectories, 
        #                                       alpha = alpha_trajectories,
        #                                       linewidth = linewidth_trajectories)
        #         elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
        #             ax[i].plot(out[2]['ndt'] + np.arange(0, out[2]['max_t'] +  out[2]['delta_t'], out[2]['delta_t'])[out[2]['trajectory'][:, 0] > -999], out[2]['trajectory'][out[2]['trajectory'] > -999], 
        #                        color = color_trajectories,
        #                        alpha = alpha_trajectories,
        #                        linewidth = linewidth_trajectories)
        #         else:
        #             ax.plot(out[2]['ndt'] + np.arange(0, out[2]['max_t'] +  out[2]['delta_t'], out[2]['delta_t'])[out[2]['trajectory'][:, 0] > -999], out[2]['trajectory'][out[2]['trajectory'] > -999],
        #                     color = color_trajectories, 
        #                     alpha = alpha_trajectories, 
        #                     linewidth = linewidth_trajectories)
                    
        #             # This part here allows for a plot in plot setup (useful for plots like the levy where we might want to show the error distribution as a little cartoon on the right)
        #             #ax_ins = ax.inset_axes([1, 0.5, 0.2, 0.2])
        #             #ax_ins.plot([0, 1, 2, 3])
        

            # Run simulations and add trajectories
        if show_trajectories == True:
            if rows > 1 and cols > 1:
                ax_tmp = ax[row_tmp, col_tmp]
            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                ax_tmp = ax[i]
            else:
                ax_tmp = ax
            
            for k in range(n_trajectories):
                out = simulator(theta = ground_truth_parameters[i, :],
                                model = model_ground_truth, 
                                n_samples = 1,
                                bin_dim = None)

                ax_tmp.plot(out[2]['ndt'] + np.arange(0, out[2]['max_t'] +  out[2]['delta_t'], out[2]['delta_t'])[out[2]['trajectory'][:, 0] > -999], out[2]['trajectory'][out[2]['trajectory'] > -999], 
                                            color = color_trajectories, 
                                            alpha = alpha_trajectories,
                                            linewidth = linewidth_trajectories)

                    

        # Run simulations and add histograms
        # True params
        if model_ground_truth is not None and ground_truth_data is None:
            out = simulator(theta = ground_truth_parameters[i, :],
                            model = model_ground_truth, 
                            n_samples = 20000,
                            bin_dim = None)
             
            tmp_true = np.concatenate([out[0], out[1]], axis = 1)
            choice_p_up_true = np.sum(tmp_true[:, 1] == 1) / tmp_true.shape[0]
        
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
        
         #ax.set_ylim(-4, 2)
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i].twinx()
        else:
            ax_tmp = ax.twinx()
        
        ax_tmp.set_ylim(-ylimit, ylimit)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]

            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)
            
            if j == (n_posterior_parameters - 1) and row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Posterior Predictive'
            else:
                tmp_label = None

            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = choice_p_up_post * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'black',
                        edgecolor = 'black',
                        zorder = -1,
                        label = tmp_label,
                        linewidth = hist_linewidth)
                
            # else:
            #     ax_tmp.hist(bins[:-1], 
            #                 bins, 
            #                 weights = choice_p_up_post * counts_2,
            #                 histtype = 'step',
            #                 alpha = 0.5, 
            #                 color = 'black',
            #                 edgecolor = 'black',
            #                 linewidth = hist_linewidth,
            #                 zorder = -1)
                        
        if model_ground_truth is not None and ground_truth_data is None:
            counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)

            if row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Ground Truth Data'
            else: 
                tmp_label = None
            
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = choice_p_up_true * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'red',
                        edgecolor = 'red',
                        zorder = -1,
                        linewidth = hist_linewidth,
                        label = tmp_label)
            
            if row_tmp == 0 an col_tmp == 0:
                ax_tmp.legend(loc = 'lower right')
            
            # else:
            #     ax_tmp.hist(bins[:-1], 
            #             bins, 
            #             weights = choice_p_up_true * counts_2,
            #             histtype = 'step',
            #             alpha = 0.5, 
            #             color = 'red',
            #             edgecolor = 'red',
            #             zorder = -1,
            #             linewidth = hist_linewidth)
        
        if ground_truth_data is not None:
            # These splits here is neither elegant nor necessary --> can represent ground_truth_data simply as a dict !
            # Wiser because either way we can have varying numbers of trials for each subject !
            if datatype == 'hierarchical' or datatype == 'condition' or datatype == 'single_subject':
                counts_2, bins = np.histogram(ground_truth_data[sorted_keys[i]][ground_truth_data[sorted_keys[i]][:, 1] == 1, 0],
                                              bins = np.linspace(0, max_t, nbins),
                                              density = True)

                choice_p_up_true_dat = np.sum(ground_truth_data[sorted_keys[i]][:, 1] == 1) / ground_truth_data[sorted_keys[i]].shape[0]
            else:
                counts_2, bins = np.histogram(ground_truth_data[i, ground_truth_data[i, :, 1] == 1, 0],
                                              bins = np.linspace(0, max_t, nbins),
                                              density = True)

                choice_p_up_true_dat = np.sum(ground_truth_data[i, :, 1] == 1) / ground_truth_data[i].shape[0]

            if row_tmp == 0 and col_tmp == 0:
                tmp_label = 'Dataset'
            else:
                tmp_label = None
            
            ax_tmp.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_true_dat * counts_2,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'blue',
                            edgecolor = 'blue',
                            zorder = -1,
                            linewidth = hist_linewidth,
                            label = tmp_label)
            
            if row_tmp == 0 and col_tmp == 0:
                ax_tmp.legend(loc = 'lower right')
            # else:
            #     ax_tmp.hist(bins[:-1], 
            #                 bins, 
            #                 weights = choice_p_up_true_dat * counts_2,
            #                 histtype = 'step',
            #                 alpha = 0.5, 
            #                 color = 'blue',
            #                 edgecolor = 'blue',
            #                 linewidth = hist_linewidth,
            #                 zorder = -1)
            
             
        #ax.invert_xaxis()
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i].twinx()
        else:
            ax_tmp = ax.twinx()
            
        ax_tmp.set_ylim(ylimit, -ylimit)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_post) * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'black',
                        edgecolor = 'black',
                        linewidth = hist_linewidth,
                        zorder = -1)
            
        if model_ground_truth is not None and ground_truth_data is None:
            counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                          bins = np.linspace(0, max_t, nbins),
                                          density = True)
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_true) * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'red',
                        edgecolor = 'red',
                        linewidth = hist_linewidth,
                        zorder = -1)
 
        if ground_truth_data is not None:
            if datatype == 'hierarchical' or datatype == 'condition' or datatype == 'single_subject':
                counts_2, bins = np.histogram(ground_truth_data[sorted_keys[i]][ground_truth_data[sorted_keys[i]][:, 1] == - 1, 0],
                                              bins = np.linspace(0, max_t, nbins),
                                              density = True)
            else:
                counts_2, bins = np.histogram(ground_truth_data[i, ground_truth_data[i, :, 1] == - 1, 0],
                                              bins = np.linspace(0, max_t, nbins),
                                              density = True)
            
            #choice_p_up_true_dat = np.sum(ground_truth_data[i, :, 1] == 1) / ground_truth_data[i].shape[0]

            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_true_dat) * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'blue',
                        edgecolor = 'blue',
                        linewidth = hist_linewidth,
                        zorder = -1)

        
        # Plot posterior samples of bounds and slopes (model)
        if show_model:
            if posterior_samples is not None:
                for j in range(n_posterior_parameters + 1):
                    
                    if j == n_posterior_parameters and model_ground_truth is not None:
                        tmp_samples = ground_truth_parameters[i, :]
                        
                        # If we supplied ground truth data --> make ground truth model blue, otherwise red
                        tmp_colors = ['red', 'blue']
                        tmp_bool = ground_truth_data is not None
                        tmp_color = tmp_colors[int(tmp_bool)]
                        tmp_alpha = 1
                        tmp_label = 'Ground Truth Model'
                    else:
                        tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.05
                        tmp_color = 'black'
                        tmp_label = None
                    
                    if j == (n_posterior_parameters - 1):
                        tmp_label = 'Model Samples'
                
                    if model_fitted == 'weibull_cdf' or model_fitted == 'weibull_cdf2' or model_fitted == 'weibull_cdf_concave':
                        b = tmp_samples[1] * bf.weibull_cdf(t = t_s, 
                                                                             alpha = tmp_samples[4],
                                                                             beta = tmp_samples[5])
                    if model_fitted == 'angle' or model_fitted == 'angle2':
                        b = np.maximum(tmp_samples[1] + bf.angle(t = t_s, 
                                                                                  theta = tmp_samples[4]), 0)
                    if model_fitted == 'ddm':
                        b = tmp_samples[1] * np.ones(t_s.shape[0])


                    start_point_tmp = - tmp_samples[1] + \
                                      (2 * tmp_samples[1] * tmp_samples[2])

                    slope_tmp = tmp_samples[0]

                    if rows > 1 and cols > 1:
                        ax_tmp = ax[row_tmp, col_tmp]
                        # ax[row_tmp, col_tmp].plot(t_s + tmp_samples[3], b, tmp_color,
                        #                           t_s + tmp_samples[3], - b, tmp_color, 
                        #                           alpha = tmp_alpha,
                        #                           zorder = 1000 + j,
                        #                           linewidth = posterior_linewidth)
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax_tmp = ax[i]
                        # ax[i].plot(t_s + tmp_samples[3], b, tmp_color,
                        #            t_s + tmp_samples[3], - b, tmp_color, 
                        #            alpha = tmp_alpha,
                        #            zorder = 1000 + j,
                        #            linewidth = posterior_linewidth,)
                    else:
                        ax_tmp = ax
                        # ax.plot(t_s + tmp_samples[3], b, tmp_color,
                        #         t_s + tmp_samples[3], - b, tmp_color, 
                        #         alpha = tmp_alpha,
                        #         zorder = 1000 + j,
                        #         linewidth = posterior_linewidth)

                    ax_tmp.plot(t_s + tmp_samples[3], b, tmp_color,
                            t_s + tmp_samples[3], - b, tmp_color, 
                            alpha = tmp_alpha,
                            zorder = 1000 + j,
                            linewidth = posterior_linewidth,
                            label = tmp_label
                            )
                    
                    if tmp_label == 'Ground Truth Model' and row_tmp == 0 and col_tmp == 0:
                        ax_tmp.legend(loc = 'upper right')

                    for m in range(len(t_s)):
                        if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                            maxid = m
                            break
                        maxid = m

                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].plot(t_s[:maxid] + tmp_samples[3],
                                                  start_point_tmp + slope_tmp * t_s[:maxid], 
                                                  c = tmp_color, 
                                                  alpha = tmp_alpha,
                                                  zorder = 1000 + j,
                                                  linewidth = posterior_linewidth,
                                                  label = tmp_label)
                        # if j == (n_posterior_parameters - 1):
                        #     ax[row_tmp, col_tmp].plot(t_s[:maxid] + tmp_samples[3],
                        #                               start_point_tmp + slope_tmp * t_s[:maxid], 
                        #                               c = 'black', 
                        #                               alpha = tmp_alpha,
                        #                               zorder = 1000 + j,
                        #                               linewidth = posterior_linewidth,
                        #                               label = 'Model Samples')
                            
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax[i].plot(t_s[:maxid] + tmp_samples[3],
                                   start_point_tmp + slope_tmp * t_s[:maxid], 
                                   c = tmp_color, 
                                   alpha = tmp_alpha,
                                   zorder = 1000 + j,
                                   linewidth = posterior_linewidth,
                                   label = tmp_label)
                        
                        # if j == (n_posterior_parameters - 1):
                        #     ax[i].plot(t_s[:maxid] + tmp_samples[3],
                        #                start_point_tmp + slope_tmp * t_s[:maxid], 
                        #                c = 'black', 
                        #                alpha = tmp_alpha,
                        #                linewidth = posterior_linewidth,
                        #                zorder = 1000 + j,
                        #                label = 'Model Samples')

                    else:
                        ax.plot(t_s[:maxid] + tmp_samples[3],
                                start_point_tmp + slope_tmp * t_s[:maxid], 
                                c = tmp_color, 
                                alpha = tmp_alpha,
                                linewidth = posterior_linewidth,
                                zorder = 1000 + j,
                                label = tmp_label)
                        # if j == (n_posterior_parameters - 1):
                        #     ax.plot(t_s[:maxid] + tmp_samples[3],
                        #             start_point_tmp + slope_tmp * t_s[:maxid], 
                        #             'black', 
                        #             alpha = tmp_alpha,
                        #             linewidth = posterior_linewidth,
                        #             zorder = 1000 + j,
                        #             label = 'Model Samples')
                                   
                    
                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].axvline(x = tmp_samples[3], 
                                                     ymin = - 2, 
                                                     ymax = 2, 
                                                     c = tmp_color, 
                                                     linestyle = '--',
                                                     linewidth = posterior_linewidth,
                                                     alpha = tmp_alpha)
                        
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax[i].axvline(x = tmp_samples[3],
                                                            ymin = - 2,
                                                            ymax = 2,
                                                            c = tmp_color,
                                                            linestyle = '--',
                                                            linewidth = posterior_linewidth,
                                                            alpha = tmp_alpha)
                    else:
                        ax.axvline(x = tmp_samples[3], 
                                   ymin = -2, 
                                   ymax = 2, 
                                   c = tmp_color, 
                                   linestyle = '--',
                                   linewidth = posterior_linewidth,
                                   alpha = tmp_alpha)
                    
                        ax.patch.set_visible(False)
               
                        
        # # If we supplied ground truth data --> make ground truth model blue, otherwise red
        # tmp_colors = ['red', 'blue']
        # tmp_bool = ground_truth_data is not None
        # tmp_color = tmp_colors[int(tmp_bool)]
                            
        # # Plot ground_truths bounds
        # if show_model and model_ground_truth is not None:
            
        #     if model_ground_truth == 'weibull_cdf' or model_ground_truth == 'weibull_cdf2' or model_ground_truth == 'weibull_cdf_concave':
        #         b = ground_truth_parameters[i, 1] * bf.weibull_cdf(t = t_s,
        #                                                  alpha = ground_truth_parameters[i, 4],
        #                                                  beta = ground_truth_parameters[i, 5])

        #     if model_ground_truth == 'angle' or model_ground_truth == 'angle2':
        #         b = np.maximum(ground_truth_parameters[i, 1] + bf.angle(t = t_s, theta = ground_truth_parameters[i, 4]), 0)

        #     if model_ground_truth == 'ddm':
        #         b = ground_truth_parameters[i, 1] * np.ones(t_s.shape[0])

        #     start_point_tmp = - ground_truth_parameters[i, 1] + \
        #                       (2 * ground_truth_parameters[i, 1] * ground_truth_parameters[i, 2])
        #     slope_tmp = ground_truth_parameters[i, 0]

        #     if rows > 1 and cols > 1:
        #         if row_tmp == 0 and col_tmp == 0:
        #             ax[row_tmp, col_tmp].plot(t_s + ground_truth_parameters[i, 3], b, tmp_color, 
        #                                       alpha = 1, 
        #                                       linewidth = gt_linewidth, 
        #                                       zorder = 1000)
        #             ax[row_tmp, col_tmp].plot(t_s + ground_truth_parameters[i, 3], -b, tmp_color, 
        #                                       alpha = 1,
        #                                       linewidth = 3,
        #                                       zorder = 1000, 
        #                                       label = 'Ground Truth Model')
        #             ax[row_tmp, col_tmp].legend(loc = 'upper right')
        #         else:
        #             ax[row_tmp, col_tmp].plot(t_s + ground_truth_parameters[i, 3], b, tmp_color, 
        #                       t_s + ground_truth_parameters[i, 3], -b, tmp_color, 
        #                       alpha = 1,
        #                       linewidth = gt_linewidth,
        #                       zorder = 1000)
                    
        #     elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
        #         if row_tmp == 0 and col_tmp == 0:
        #             ax[i].plot(t_s + ground_truth_parameters[i, 3], b, tmp_color, 
        #                                       alpha = 1, 
        #                                       linewidth = gt_linewidth, 
        #                                       zorder = 1000)
        #             ax[i].plot(t_s + ground_truth_parameters[i, 3], -b, tmp_color, 
        #                                       alpha = 1,
        #                                       linewidth = gt_linewidth,
        #                                       zorder = 1000, 
        #                                       label = 'Ground Truth Model')
        #             ax[i].legend(loc = 'upper right')
        #         else:
        #             ax[i].plot(t_s + ground_truth_parameters[i, 3], b, tmp_color, 
        #                       t_s + ground_truth_parameters[i, 3], -b, tmp_color, 
        #                       alpha = 1,
        #                       linewidth = gt_linewidth,
        #                       zorder = 1000)
        #     else:
        #         ax.plot(t_s + ground_truth_parameters[i, 3], b, tmp_color, 
        #                 alpha = 1, 
        #                 linewidth = gt_linewidth, 
        #                 zorder = 1000)
        #         ax.plot(t_s + ground_truth_parameters[i, 3], -b, tmp_color, 
        #                 alpha = 1,
        #                 linewidth = gt_linewidth,
        #                 zorder = 1000,
        #                 label = 'Ground Truth Model')
        #         #print('passed through legend part')
        #         #print(row_tmp)
        #         #print(col_tmp)
        #         ax.legend(loc = 'upper right')

        #     # Ground truth slope:
        #     for m in range(len(t_s)):
        #         if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
        #             maxid = m
        #             break
        #         maxid = m

        #     # print('maxid', maxid)
        #     if rows > 1 and cols > 1:
        #         ax[row_tmp, col_tmp].plot(t_s[:maxid] + ground_truth_parameters[i, 3], 
        #                                   start_point_tmp + slope_tmp * t_s[:maxid], 
        #                                   tmp_color, 
        #                                   alpha = 1, 
        #                                   linewidth = gt_linewidth, 
        #                                   zorder = 1000)

        #         ax[row_tmp, col_tmp].set_zorder(ax_tmp.get_zorder() + 1)
        #         ax[row_tmp, col_tmp].patch.set_visible(False)
        #     elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
        #         ax[i].plot(t_s[:maxid] + ground_truth_parameters[i, 3], 
        #                                   start_point_tmp + slope_tmp * t_s[:maxid], 
        #                                   tmp_color, 
        #                                   alpha = 1, 
        #                                   linewidth = gt_linewidth, 
        #                                   zorder = 1000)

        #         ax[i].set_zorder(ax_tmp.get_zorder() + 1)
        #         ax[i].patch.set_visible(False)
        #     else:
        #         ax.plot(t_s[:maxid] + ground_truth_parameters[i, 3], 
        #                 start_point_tmp + slope_tmp * t_s[:maxid], 
        #                 tmp_color, 
        #                 alpha = 1, 
        #                 linewidth = gt_linewidth, 
        #                 zorder = 1000)

        #         ax.set_zorder(ax_tmp.get_zorder() + 1)
        #         ax.patch.set_visible(False)
               
        # Set plot title
        title_tmp = ''
        
        if model_ground_truth is not None:
            for k in range(len(ax_titles)):
                title_tmp += ax_titles[k] + ': '
                title_tmp += str(round(ground_truth_parameters[i, k], 2)) + ', '

        if rows > 1 and cols > 1:
            if row_tmp == rows:
                ax[row_tmp, col_tmp].set_xlabel('rt', 
                                                 fontsize = 20);
            ax[row_tmp, col_tmp].set_ylabel('', 
                                            fontsize = 20);


            ax[row_tmp, col_tmp].set_title(title_tmp,
                                           fontsize = 24)
            ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 20)
            ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 20)

            # Some extra styling:
            if model_ground_truth is not None:
                if show_model:
                    ax[row_tmp, col_tmp].axvline(x = ground_truth_parameters[i, 3], ymin = -2, ymax = 2, c = tmp_color, linestyle = '--')
                ax[row_tmp, col_tmp].axhline(y = 0, xmin = 0, xmax = ground_truth_parameters[i, 3] / max_t, c = tmp_color,  linestyle = '--')
        
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            if row_tmp == rows:
                ax[i].set_xlabel('rt', 
                                 fontsize = 20);
            ax[i].set_ylabel('', 
                             fontsize = 20);

            ax[i].set_title(title_tmp,
                            fontsize = 24)
            
            ax[i].tick_params(axis = 'y', size = 20)
            ax[i].tick_params(axis = 'x', size = 20)

            # Some extra styling:
            if model_ground_truth is not None:
                if show_model:
                    ax[i].axvline(x = ground_truth_parameters[i, 3], ymin = -2, ymax = 2, c = tmp_color, linestyle = '--')
                ax[i].axhline(y = 0, xmin = 0, xmax = ground_truth_parameters[i, 3] / max_t, c = tmp_color,  linestyle = '--')
        
        else:
            if row_tmp == rows:
                ax.set_xlabel('rt', 
                              fontsize = 20);
            ax.set_ylabel('', 
                          fontsize = 20);

            ax.set_title(title_tmp,
                         fontsize = 24)

            ax.tick_params(axis = 'y', size = 20)
            ax.tick_params(axis = 'x', size = 20)

            # Some extra styling:
            if model_ground_truth is not None:
                if show_model:
                    ax.axvline(x = ground_truth_parameters[i, 3], ymin = -2, ymax = 2, c = tmp_color, linestyle = '--')
                ax.axhline(y = 0, xmin = 0, xmax = ground_truth_parameters[i, 3] / max_t, c = tmp_color,  linestyle = '--')

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


# STRUCTURE
# EXPECT A HDDM MODEL
# TRANSFORM TRACES INTO USABLE FORM
# PRODUCE PLOT