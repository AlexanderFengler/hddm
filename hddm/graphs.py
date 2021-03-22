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
               scale_y = 1.0,
               delta_t_graph = 0.01):
    
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
    elif ground_truth_data is not None and datatype == 'hierarchical':
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

    elif ground_truth_data is not None and datatype == 'dict':
        sorted_keys = np.sort(list(ground_truth_data.keys()))

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
            if posterior_samples is not None:
                for j in range(n_posterior_parameters + 1):
                    tmp_label = ""
                    if j == (n_posterior_parameters - 1):
                        tmp_label = 'Model Samples'
                        tmp_model = model_fitted
                        tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.5
                        tmp_color = 'black'
                    elif j == n_posterior_parameters and model_ground_truth is not None:
                        tmp_samples = ground_truth_parameters[i, :]
                        tmp_model = model_ground_truth
                        
                        # If we supplied ground truth data --> make ground truth model blue, otherwise red
                        tmp_colors = ['red', 'blue']
                        tmp_bool = ground_truth_data is not None
                        tmp_color = tmp_colors[int(tmp_bool)]
                        tmp_alpha = 1
                        tmp_label = 'Ground Truth Model'
                    elif j == n_posterior_parameters and model_ground_truth == None:
                        break
                    else:
                        tmp_model = model_fitted
                        tmp_samples = posterior_samples[i, idx[j], :]
                        tmp_alpha = 0.05
                        tmp_color = 'black'
                        tmp_label = None

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
                                    cartoon = True,
                                    delta_t = delta_t_graph,
                                    bin_dim = None)
                    
                    tmp_traj = out[2]['trajectory']
                    maxid = np.minimum(np.argmax(np.where(tmp_traj > - 999)), t_s.shape[0])

                    ax_tmp.plot(t_s + tmp_samples[model_config[tmp_model]['params'].index('t')], b, tmp_color,
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = posterior_linewidth,
                                label = tmp_label,
                                )

                    ax_tmp.plot(t_s + tmp_samples[model_config[tmp_model]['params'].index('t')], -b, tmp_color, 
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = posterior_linewidth,
                                )

                    ax_tmp.plot(t_s[:maxid] + tmp_samples[model_config[tmp_model]['params'].index('t')],
                                tmp_traj[:maxid],
                                c = tmp_color, 
                                alpha = tmp_alpha,
                                zorder = 1000 + j,
                                linewidth = posterior_linewidth) # TOOK AWAY LABEL

                    ax_tmp.axvline(x = tmp_samples[model_config[tmp_model]['params'].index('t')], # this should identify the index of ndt directly via model config !
                                   ymin = - ylimit, 
                                   ymax = ylimit, 
                                   c = tmp_color, 
                                   linestyle = '--',
                                   linewidth = posterior_linewidth,
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
                ax_tmp.axvline(x = ground_truth_parameters[i, model_config[model_ground_truth]['params'].index('t')], ymin = - 2, ymax = 2, c = tmp_color, linestyle = '--')
            ax_tmp.axhline(y = 0, xmin = 0, xmax = ground_truth_parameters[i, model_config[model_ground_truth]['params'].index('t')] / max_t, c = tmp_color,  linestyle = '--')
        
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

    # def model_plot(posterior_samples = None,
    #            ground_truth_parameters = None,
    #            ground_truth_data = None,
    #            model_ground_truth = 'weibull_cdf',
    #            model_fitted = 'angle',
    #            input_is_hddm_trace = False,
    #            datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition' # data structure
    #            condition_column = 'condition', # data structure
    #            n_plots = 4, 
    #            n_posterior_parameters = 500,
    #            n_simulations_per_parameter = 10,
    #            cols = 3, # styling
    #            max_t = 5, # styling
    #            show_model = True, # styling
    #            show_trajectories = False, # styling
    #            n_trajectories = 10,
    #            color_trajectories = 'blue',
    #            alpha_trajectories = 0.2,
    #            linewidth_trajectories = 1.0,
    #            ylimit = 2, # styling
    #            posterior_linewidth = 3, # styling
    #            gt_linewidth = 3, # styling
    #            hist_linewidth = 3, # styling
    #            bin_size = 0.025, # styling
    #            save = False,
    #            scale_x = 1.0,
    #            scale_y = 1.0,
    #            delta_t_graph = 0.01):

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
    
    
#                 counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
#                                           bins = np.linspace(0, max_t, 100),
#                                           density = True)

    if save == True:
        pass
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['pdf.fonttype'] = 42
        #matplotlib.rcParams['svg.fonttype'] = 'none'
    
    if model_ground_truth is None and ground_truth_data is None and posterior_samples is None:
        return 'No ground truth model was supplied, no dataset was supplied and no posterior sample was supplied. Nothin to plot' 
    
    # Take care of ground_truth_data
    if ground_truth_data is not None and datatype == 'hierarchical':
        gt_dat_dict = dict()
        for i in np.sort(np.unique(ground_truth_data['subj_idx'])):
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data['subj_idx'] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict
        # print('Supplying ground truth data not yet implemented for hierarchical datasets')
        
    if ground_truth_data is not None and datatype == 'condition':
        gt_dat_dict = dict()
        for i in np.sort(np.unique(ground_truth_data[condition_column])):
            gt_dat_dict[i] = ground_truth_data.loc[ground_truth_data[condition_column] == i][['rt', 'response']]
            gt_dat_dict[i].loc[gt_dat_dict[i]['response'] == 0,  'response'] = - 1
            gt_dat_dict[i] = gt_dat_dict[i].values
        ground_truth_data = gt_dat_dict

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
            #print(posterior_samples)
            #n_plots = posterior_samples.shape[0]
            
    if n_plots == 1:
        rows = 1
        cols = 1
    
    nbins = int((2 * max_t) / bin_size)
        # Taking care of special case with 1 plot
    if n_plots == 1:
        if model_ground_truth is not None:
            ground_truth_parameters = np.expand_dims(ground_truth_parameters, 0)
        if posterior_samples is not None:
            posterior_samples = np.expand_dims(posterior_samples, 0)
        if ground_truth_data is not None:
            ground_truth_data = np.expand_dims(ground_truth_data, 0)
         
    
#     matplotlib.rcParams['text.usetex'] = True
#     #matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['svg.fonttype'] = 'none'
    
    rows = int(np.ceil(n_plots / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    # fig, ax = plt.subplots(rows, cols, 
    #                        figsize = (20 * scale_x, 20 * rows * scale_y), 
    #                        sharex = False, 
    #                        sharey = False)
    

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20 * scale_x, 20 * rows * scale_y), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Posterior Predictive: ' + model_fitted.upper(),
                 fontsize = 24)
    
    sns.despine(right = True)
#     tmp_simulator = simulator(model = model, 
#                               n_samples = 20000,
#                               bin_dim = None)
    
    for i in range(n_plots):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        post_tmp = np.zeros((n_posterior_parameters * n_simulations_per_parameter, 2))
        idx = np.random.choice(posterior_samples.shape[1], 
                               size = n_posterior_parameters, 
                               replace = False)

        # Run Model simulations for posterior samples
        for j in range(n_posterior_parameters):
            out = simulator(theta = posterior_samples[i, idx[j], :], 
                            model = model_fitted,
                            n_samples = n_simulations_per_parameter,
                            bin_dim = None)
          
            post_tmp[(n_simulations_per_parameter * j):(n_simulations_per_parameter * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
        
        # Run Model simulations for true parameters
        # Supply data too !
        if model_ground_truth is not None:
            out = simulator(theta = ground_truth_parameters[i, :],
                            model = model_ground_truth,
                            n_samples = 20000,
                            bin_dim = None)
  
            gt_tmp = np.concatenate([out[0], out[1]], axis = 1)
            gt_color = 'red'
            #print('passed through')
        else:
            gt_tmp = ground_truth_data[i]
            gt_color = 'blue'

        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp]
        
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i]
        
        else:
            ax_tmp = ax
        
        # ACTUAL PLOTTING
        # if rows > 1 and cols > 1:
        ax[row_tmp, col_tmp].hist(post_tmp[:, 0] * post_tmp[:, 1], 
                                  bins = np.linspace(- max_t, max_t, nbins), #50, # kde = False, # rug = False, 
                                  alpha =  1, 
                                  color = 'black',
                                  histtype = 'step', 
                                  density = 1, 
                                  edgecolor = 'black',
                                  linewidth = hist_linewidth
                                  )
        
#             sns.histplot(post_tmp[:, 0] * post_tmp[:, 1], 
#                          bins = np.linspace(-max_t, max_t, nbins), #50, 
#                          kde = False, # rug = False, 
#                          hist_kws = {'alpha': 1, 
#                                      'color': 'black',
#                                      'histtype': 'step', 
#                                      'density': 1, 
#                                      'edgecolor': 'black',
#                                      'linewidth': hist_linewidth},
#                          ax = ax[row_tmp, col_tmp]);

        ax[row_tmp, col_tmp].hist(gt_tmp[:, 0] * gt_tmp[:, 1], 
                                  alpha = 0.5, 
                                  color = gt_color, 
                                  density = 1, 
                                  edgecolor = gt_color,  
                                  histtype = 'step',
                                  linewidth = hist_linewidth, 
                                  bins = np.linspace(-max_t, max_t, nbins), #50, 
                                  # kde = False, #rug = False,
                                  )
        
        # elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            
        #     ax[i].hist(post_tmp[:, 0] * post_tmp[:, 1], 
        #                bins = np.linspace(-max_t, max_t, nbins), #50, # kde = False, #rug = False, 
        #                alpha = 1, 
        #                color = 'black',
        #                histtype = 'step', 
        #                density = 1, 
        #                edgecolor = 'black',
        #                linewidth = hist_linewidth
        #                )

        #     ax[i].hist(gt_tmp[:, 0] * gt_tmp[:, 1], 
        #                alpha = 0.5, 
        #                color = gt_color, 
        #                density = 1, 
        #                edgecolor = gt_color, 
        #                histtype = 'step',
        #                linewidth = hist_linewidth, 
        #                bins = np.linspace(-max_t, max_t, nbins), #50, # kde = False, #rug = False,
        #                )
            
        # else:
            
        #     ax.hist(post_tmp[:, 0] * post_tmp[:, 1], 
        #             bins = np.linspace(-max_t, max_t, nbins), #50, # kde = False, #rug = False,
        #             alpha = 1, 
        #             color = 'black', 
        #             histtype = 'step', 
        #             density = 1, 
        #             edgecolor = 'black',
        #             linewidth = hist_linewidth,
        #             );
            
        #     ax.hist(gt_tmp[:, 0] * gt_tmp[:, 1], 
        #             alpha = 0.5, 
        #             color = gt_color, 
        #             density = 1,
        #             edgecolor = gt_color,  
        #             histtype = 'step',
        #             linewidth = hist_linewidth, 
        #             bins = np.linspace(-max_t, max_t, nbins), #50, # kde = False, #rug = False,
        #             )
        
#         if rows > 1 and cols > 1:
#             ax[row_tmp, col_tmp].set_xlim(0, max_t)
#             ax[row_tmp, col_tmp].set_ylim(-ylimit, ylimit)
#         elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
#             ax[i].set_xlim(0, max_t)
#             ax[i].set_ylim(-ylimit, ylimit)
#         else:
#             ax.set_xlim(0, max_t)
#             ax.set_ylim(-ylimit, ylimit)
        
        # if rows > 1 and cols > 1:
        #     ax_tmp = ax[row_tmp, col_tmp]
        # elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
        #     ax_tmp = ax[i]
        # else:
        #     ax_tmp = ax

        # EXTRA STYLING    
        ax_tmp.set_xlim(- xlimit, xlimit)
            
        if row_tmp == 0 and col_tmp == 0:
            if model_ground_truth is not None:
                label_0 = 'GROUND TRUTH'
            else:
                label_0 = 'DATA'
            ax_tmp.legend(labels = ['POSTERIOR PREDICTIVE', label_0], 
                          fontsize = 12, 
                          loc = 'upper right')
            
#             ax[row_tmp, col_tmp].legend(labels = [model_fitted, 'posterior'], 
#                                         fontsize = 12, loc = 'upper right')
        
        if row_tmp == (rows - 1):
            ax_tmp.set_xlabel('RT', 
                              fontsize = 24)
#             ax[row_tmp, col_tmp].set_xlabel('RT', 
#                                             fontsize = 14);
        
        if col_tmp == 0:
            ax_tmp.set_ylabel('', 
                              fontsize = 24)
#             ax[row_tmp, col_tmp].set_ylabel('Density', 
#                                             fontsize = 14);
        
#         ax[row_tmp, col_tmp].set_title(ax_titles[i],
#                                        fontsize = 16)

        ax_tmp.tick_params(axis = 'y', size = 22)
        ax_tmp.tick_params(axis = 'x', size = 22)
        
#         ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 12)
#         ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 12)
        
        # THIS MAY NOT BE NECESSARY ?
        # if rows > 1 and cols > 1:
        #     ax[row_tmp, col_tmp] = ax_tmp
        # elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
        #     ax[i] = ax_tmp
        # else:
        #     ax = ax_tmp
        
    if rows > 1 and cols > 1:
        for i in range(n_plots, rows * cols, 1):
            row_tmp = int(np.floor(i / cols))
            col_tmp = i - (cols * row_tmp)
            ax[row_tmp, col_tmp].axis('off')
            
            
    if save == True:
        plt.savefig('figures/' + 'posterior_predictive_plot_' + model_ground_truth + '_' + datatype + '.svg',
                    format = 'svg', 
                    transparent = True,
                    frameon = False)
        plt.close()

    return plt.show()

# STRUCTURE
# EXPECT A HDDM MODEL
# TRANSFORM TRACES INTO USABLE FORM
# PRODUCE PLOT