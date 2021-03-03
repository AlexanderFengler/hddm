import pandas as pd
import numpy as np
from copy import deepcopy
#import re
import argparse
import sys
import pickle

from hddm.simulators.basic_simulator import *

# Helper
def hddm_preprocess(simulator_data = None, subj_id = 'none'):
    
    df = pd.DataFrame(simulator_data[0].astype(np.double), columns = ['rt'])
    df['response'] = simulator_data[1].astype(int)
    df['nn_response'] = df['response']
    df.loc[df['response'] == -1.0, 'response'] = 0.0
    df['subj_idx'] = subj_id
    return df


def str_to_num(string = '', n_digits = 3):
    new_str = ''
    leading = 1
    for digit in range(n_digits):
        if string[digit] == '0' and leading and (digit < n_digits - 1):
            pass
        else:
            new_str += string[digit]
            leading = 0
    return int(new_str)

def num_to_str(num = 0, n_digits = 3):
    new_str = ''
    for i in range(n_digits - 1, -1, -1):
        if num < np.power(10, i):
            new_str += '0'
    if num != 0:
        new_str += str(num)
    return new_str

def _pad_subj_id(in_str):
    # Make subj ids have three digits by prepending 0s if necessary
    stridx = in_str.find('.') # get index of 'subj.' substring
    subj_idx_len = len(in_str[(stridx + len('.')):]) # check how many letters remain after 'subj.' is enocuntered
    out_str = ''
    prefix_str = ''
    for i in range(3 - subj_idx_len):
        prefix_str += '0' # add zeros to pad subject id to have three digits

    out_str = in_str[:stridx + len('.')] + prefix_str + in_str[stridx + len('.'):] #   
    # print(out_str)
    return out_str

# -------------------------------------------------------------------------------------

# Dataset generators
def simulator_single_subject(parameters = [0, 0, 0],
                             model = 'angle',
                             n_samples = 1000,
                             delta_t = 0.001,
                             max_t = 20,
                             bin_dim = None,
                             bin_pointwise = False):
    
    x = simulator(theta = parameters,
                  model = model,
                  n_samples = n_samples,
                  delta_t = delta_t,
                  max_t = max_t,
                  bin_dim = bin_dim,
                  bin_pointwise = bin_pointwise)
    
    return hddm_preprocess(x)
    
def simulator_stimcoding(model = 'angle',
                         split_by = 'v',
                         decision_criterion = 0.0,
                         n_samples_by_condition = 1000):
    
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (2, 1))
    
              
    #len(model_config[model]['params']                   
    #print(param_base)
    gt = {}
    for i in range(2):
        id_tmp = model_config[model]['params'].index(split_by)
        
        if i == 0:
#             param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
#                                                       high = model_config[model]['param_bounds'][1][id_tmp])
            gt[split_by] = param_base[i, id_tmp]
            gt['decision_criterion'] = decision_criterion
            if split_by == 'v':
                param_base[i, id_tmp] = decision_criterion + param_base[i, id_tmp]
            
        if i == 1:
            if split_by == 'v':
                param_base[i, id_tmp] = decision_criterion - param_base[i, id_tmp]
            if split_by == 'z':
                param_base[i, id_tmp] = 1 - param_base[i, id_tmp]
            
    #print(param_base)
    dataframes = []
    for i in range(2):
        sim_out = simulator(param_base[i, :], 
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = None)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i + 1))
    
    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns = {'subj_idx': "stim"})
    # print(param_base.shape)
    return (data_out, gt, param_base)

def simulator_condition_effects(n_conditions = 4, 
                                n_samples_by_condition = 1000,
                                condition_effect_on_param = [0], 
                                model = 'angle',
                                ):
     
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                            high = model_config[model]['param_bounds'][1], 
                                            size = (1, len(model_config[model]['params']))),
                                            (n_conditions, 1))
                        
    #len(model_config[model]['params']                   
    #print(param_base)
    gt = {}
    for i in range(n_conditions):
        for c_eff in condition_effect_on_param:
            id_tmp = model_config[model]['params'].index(c_eff)
            #print(id_tmp)
            #print(model_config[model]['param_bounds'][0])
            param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
                                                    high = model_config[model]['param_bounds'][1][id_tmp])
            gt[c_eff + '(' + str(i) + ')'] = param_base[i, id_tmp]
    
    for param in model_config[model]['params']:
        if param in condition_effect_on_param:
            pass
        else:
            id_tmp = model_config[model]['params'].index(param)
            gt[param] = param_base[0, id_tmp]
            
    #print(param_base)
    dataframes = []
    for i in range(n_conditions):
        sim_out = simulator(param_base[i, :], 
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = None)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i))
    
    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns = {'subj_idx': "condition"})
    # print(param_base.shape)
    return (data_out, gt, param_base)

def simulator_covariate(dependent_params = ['v'],
                        model = 'angle',
                        n_samples = 1000,
                        beta = 0.1,
                        subj_id = 'none'):
    
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (n_samples, 1))
    
    # TD: Be more clever about covariate magnitude (maybe supply?)
    tmp_covariate_by_sample = np.random.uniform(low = - 1.0, high = 1.0, size = n_samples)
    for covariate in dependent_params:
        id_tmp = model_config[model]['params'].index(covariate)
        param_base[:, id_tmp] = param_base[:, id_tmp] + (beta * tmp_covariate_by_sample)
    
    rts = []
    choices = []
    for i in range(n_samples):
        sim_out = simulator(param_base[i, :],
                            model = model,
                            n_samples = 1,
                            bin_dim = None)
        
        rts.append(sim_out[0])
        choices.append(sim_out[1])
    
    rts = np.squeeze(np.stack(rts, axis = 0))
    choices = np.squeeze(np.stack(choices, axis = 0))
    
    data = hddm_preprocess([rts, choices], subj_id)
    data['BOLD'] = tmp_covariate_by_sample
    
    return (data, param_base, beta)

def simulator_hierarchical(n_subjects = 5,
                           n_samples_by_subject = 10,
                           model = 'angle'):

    param_ranges_half = (np.array(model_config[model]['param_bounds'][1]) - np.array(model_config[model]['param_bounds'][0])) / 2
    
    global_stds = np.random.uniform(low = 0.01, 
                                    high = param_ranges_half / 6,
                                    size = (1, len(model_config[model]['param_bounds'][0])))
    
    global_means = np.random.uniform(low = model_config[model]['param_bounds'][0],
                                     high = model_config[model]['param_bounds'][1],
                                     size = (1, len(model_config[model]['param_bounds'][0])))
                                    
    
    dataframes = []
    subject_parameters = np.zeros((n_subjects, 
                                   len(model_config[model]['param_bounds'][0])))
    gt = {}
    
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        gt[param] = global_means[0, id_tmp]
        gt[param + '_std'] = global_stds[0, id_tmp]
    
    for i in range(n_subjects):
        subj_id = num_to_str(i)
        # Get subject parameters
        a = (model_config[model]['param_bounds'][0] - global_means[0, :]) / global_stds[0, :]
        b = (model_config[model]['param_bounds'][1] - global_means[0, :]) / global_stds[0, :]
        
        subject_parameters[i, :] = np.float32(global_means[0, :] + (truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[0, :]))
        
        sim_out = simulator(subject_parameters[i, :],
                            model = model,
                            n_samples = n_samples_by_subject,
                            bin_dim = None)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, 
                                          subj_id = subj_id))
        
        for param in model_config[model]['params']:
            id_tmp = model_config[model]['params'].index(param)
            gt[param + '_subj.' + subj_id] = subject_parameters[i, id_tmp]
        
    data_out = pd.concat(dataframes)
    
    return (data_out, gt, subject_parameters)   