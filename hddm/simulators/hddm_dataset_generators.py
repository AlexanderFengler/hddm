import pandas as pd
import numpy as np
from copy import deepcopy
#import re
import argparse
import sys
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import truncnorm

from hddm.simulators.basic_simulator import *

# Helper
def hddm_preprocess(simulator_data = None, subj_id = 'none', keep_negative_responses = False):
    
    # Define dataframe if simulator output is normal (comes out as list tuple [rts, choices, metadata])
    if len(simulator_data) == 3:
        #print('Simulator data: ')
        #print(simulator_data[0])
        #print(simulator_data[0].shape)
        df = pd.DataFrame(simulator_data[0].astype(np.double), columns = ['rt'])
        df['response'] = simulator_data[1].astype(int)
    # Define dataframe if simulator output is binned pointwise (comes out as tuple [np.array, metadata])
    
    # I think this part is never called !
    if len(simulator_data) == 2:
        df = pd.DataFrame(simulator_data[0][:, 0], columns = ['rt'])
        df['response'] = simulator_data[0][:, 1].astype(int)

    #df['nn_response'] = df['response']
    if not keep_negative_responses:
        df.loc[df['response'] == -1.0, 'response'] = 0.0
    
    df['subj_idx'] = subj_id
    return df


def _str_to_num(string = '', n_digits = 3):
    new_str = ''
    leading = 1
    for digit in range(n_digits):
        if string[digit] == '0' and leading and (digit < n_digits - 1):
            pass
        else:
            new_str += string[digit]
            leading = 0
    return int(new_str)

def _num_to_str(num = 0, n_digits = 3):
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
    """Generate a hddm-ready dataset from a single set of parameters

    :Arguments:
        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        n_samples: int <default=1000>
            Number of samples to simulate.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: panda.DataFrame holding a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
    """
    
    x = simulator(theta = parameters,
                  model = model,
                  n_samples = n_samples,
                  delta_t = delta_t,
                  max_t = max_t,
                  bin_dim = bin_dim,
                  bin_pointwise = bin_pointwise)
    
    return hddm_preprocess(x)

# TD: DIDN'T GO OVER THIS ONE YET !
def simulator_stimcoding(model = 'angle',
                         split_by = 'v',
                         decision_criterion = 0.0,
                         n_samples_by_condition = 1000,
                         delta_t = 0.001,
                         prespecified_params = {},
                         bin_pointwise = True,
                         bin_dim = None,
                         max_t = 20.0):

    """Generate a dataset as expected by Hddmstimcoding. Essentially it is a specific way to parameterize two condition data.

    :Arguments:
        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        split_by: str <default='v'>
            You can split by 'v' or 'z'. If splitting by 'v' one condition's v_0 = decision_criterion + 'v', the other 
            condition's v_1 = decision_criterion - 'v'.
            Respectively for 'z', 'z_0' = 'z' and 'z_1' = 1 - 'z'.
        decision_criterion: float <default=0.0>
            Parameter that can be treated as the 'bias part' of the slope, in case we split_by 'v'.
        n_samples_by_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        prespecified_params: dict <default = {}>
            A dictionary with parameter names keys. Values are list of either length 1, or length equal to the number of conditions (here 2).
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Return: panda.DataFrame holding a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
    """
    
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (2, 1))

    # Fill in prespecified parameters if supplied
    if prespecified_params is not None:
        if type(prespecified_params) == dict:
            for param in prespecified_params:
                id_tmp = model_config[model]['params'].index(param)
                param_base[:, id_tmp] = prespecified_params[param]
        else:
            print('prespecified_params is not supplied as a dictionary, please reformat the input')
            return

    
    if type(split_by) == list:
        pass
    elif type(split_by) == str:
        split_by = [split_by]
    else:
        print('Can not recognize data-type of argument: split_by, provided neither a list nor a string')
        return
    gt = {}

    for i in range(2):
        
        if i == 0:
#             param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
#                                                       high = model_config[model]['param_bounds'][1][id_tmp])
            if 'v' in split_by:
                id_tmp = model_config[model]['params'].index('v')
                param_base[i, id_tmp] = decision_criterion + param_base[i, id_tmp]
                gt['v'] = param_base[i, id_tmp]
                gt['decision_criterion'] = decision_criterion
   
        if i == 1:
            
            if 'v' in split_by:
                id_tmp = model_config[model]['params'].index('v')
                param_base[i, id_tmp] = decision_criterion - param_base[i, id_tmp]
            if 'z' in split_by:
                id_tmp = model_config[model]['params'].index('z')
                param_base[i, id_tmp] = 1 - param_base[i, id_tmp]
            
    #print(param_base)
    dataframes = []
    for i in range(2):
        
        sim_out = simulator(param_base[i, :], 
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t)

        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i + 1))
    
    data_out = pd.concat(dataframes)
    data_out = data_out.rename(columns = {'subj_idx': "stim"})
    # print(param_base.shape)
    return (data_out, gt, param_base)

def simulator_condition_effects(n_conditions = 4,
                                n_samples_by_condition = 1000,
                                condition_effect_on_param = None,
                                prespecified_params = None,
                                model = 'angle',
                                bin_dim = None,
                                bin_pointwise = False,
                                max_t = 20.0,
                                delta_t = 0.001,
                                ):

    """ Generate a dataset with multiple conditions.

    :Arguments:
        n_conditions: int <default=4>


        parameters: list or numpy array
            Model parameters with which to simulate.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        n_samples_by_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        condition_effect_on_param: list of strings <default=None>
            List containing the parameters which will be affected by the condition.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        prespecified_params: dict <default = {}>
            A dictionary with parameter names keys. Values are list of either length 1, or length equal to the number of conditions.
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
        panda.DataFrame: Holds a 'reaction time' column and a 'response' column. Ready to be fit with hddm.
    """

    # Get list of keys in prespecified_params and return if it is not a dict when it is in fact not None
    if prespecified_params is not None:
        if type(prespecified_params) == dict:
            prespecified_params_names = list(prespecified_params.keys())
        else:
            print('prespecified_params is not a dictionary')
            return
               
    
    # Randomly assign values to every parameter and then copy across rows = number of conditions
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                            high = model_config[model]['param_bounds'][1], 
                                            size = (1, len(model_config[model]['params']))),
                                            (n_conditions, 1))
    
    
         
    # Reassign parameters according to the information in prespecified params and condition_effect_on_param
    gt = {}

    # Loop over valid model parameters
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        
        # Check if parameter is affected by condition
        if param in condition_effect_on_param:
            
            # If parameter is affected by condition we loop over conditions
            for i in range(n_conditions):
                # Assign randomly
                param_base[i, id_tmp] = np.random.uniform(low = model_config[model]['param_bounds'][0][id_tmp], 
                                                          high = model_config[model]['param_bounds'][1][id_tmp])
                
                # But if we actually specified it for each condition
                if prespecified_params is not None:
                    if param in prespecified_params_names:
                        # We assign it from prespecified dictionary
                        param_base[i, id_tmp] = prespecified_params[param][i] 
                
                # Add info to ground truth dictionary
                gt[param + '(' + str(i) + ')'] = param_base[i, id_tmp]
        
        # If the parameter is not affected by condition     
        else:
            # But prespecified
            if prespecified_params is not None:
                
                if param in prespecified_params_names:
                    # We assign prespecifided param
                    tmp_param = prespecified_params[param]
                    param_base[:, id_tmp] = tmp_param   

            # If it wasn't prespecified we just keep the random assignment that was generated above before the loops
            gt[param] = param_base[0, id_tmp]
    
    dataframes = []
    for i in range(n_conditions):
        sim_out = simulator(param_base[i, :],
                            model = model, 
                            n_samples = n_samples_by_condition,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t,
                            delta_t = delta_t)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, subj_id = i))
    
    data_out = pd.concat(dataframes)
    
    # Change 'subj_idx' column name to 'condition' ('subj_idx' is assigned automatically by hddm_preprocess() function)
    data_out = data_out.rename(columns = {'subj_idx': "condition"})
    data_out['subj_idx'] = 0
    data_out.reset_index(drop = True, inplace = True)

    if bin_pointwise:
        data_out['rt'] = data_out['rt'].astype(np.int_)
        data_out['response'] = data_out['response'].astype(np.int_)
        #data_out['nn_response'] = data_out['nn_response'].astype(np.int_)

    return (data_out, gt, param_base)

def simulator_covariate(dependent_params = ['v'],
                        model = 'angle',
                        n_samples = 1000,
                        betas = {'v': 0.1},
                        covariate_magnitudes = {'v': 1.0},
                        prespecified_params = None,
                        subj_id = 'none',
                        bin_dim = None, 
                        bin_pointwise = False,
                        max_t = 20.0,
                        delta_t = 0.001,
                        ):

    """ Generate a dataset which includes covariates. Some parameters are now a function (regression) covariates.

    :Arguments:
        dependent_params: list of strings <default=['v']>
            Parameters which will be treated as a deterministic function of a covariate
        prespecified_params: list or numpy array
            A list or numpy array of parameters to prespecify. These parameters are not functions of covariates.
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        betas: dict <default={'v': 0.1}>
            Ground truth regression betas for the parameters which are functions of covariates.
        covariates_magnitudes: dict <default={'v': 1.0}>
            A dict which holds magnitudes of the covariate vectors (value), by for each parameters (key).
        n_samples_by_condition: int <default=1000>
            Number of samples to simulate per condition (here 2 condition by design).
        condition_effect_on_param: list of strings <default=None>
            List containing the parameters which will be affected by the condition.
        subj_id: str <default='none'>
            Hddm expects a subject column in the dataset. This supplies a specific label if so desired.
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
        (panda.DataFrame, dict): The Dataframe holds a 'reaction time' column, a 'response' column and a 'BOLD' column (for the covariate). The dictionary holds the groundtruth parameter (values) and parameter names (keys).
                                 Ready to be fit with hddm.
    
    """

    if betas == None:
        betas = {}
    if covariate_magnitudes == None:
        covariate_magnitudes = {}
    if len(dependent_params) < 1:
        print('If there are no dependent variables, no need for the simulator which includes covariates')
        return

    # sanity check that prespecified parameters do not clash with parameters that are supposed to derive from trial-wise regression
    if prespecified_params is not None:
        for param in prespecified_params:
            if param in covariate_magnitudes.keys() or param in betas.keys():
                'Parameters that have covariates are Prespecified, this should not be intented'
                return

    # Fill parameter matrix
    param_base = np.tile(np.random.uniform(low = model_config[model]['param_bounds'][0],
                                           high = model_config[model]['param_bounds'][1], 
                                           size = (1, len(model_config[model]['params']))),
                                           (n_samples, 1))

    # Adjust any parameters that where prespecified
    if prespecified_params is not None:
        for param in prespecified_params.keys():
            id_tmp = model_config[model]['params'].index(param)
            param_base[:, id_tmp] = prespecified_params[param]

    # TD: Be more clever about covariate magnitude (maybe supply?)
    # Parameters that have a
    for covariate in dependent_params:
        id_tmp = model_config[model]['params'].index(covariate)

        if covariate in covariate_magnitudes.keys():
            tmp_covariate_by_sample = np.random.uniform(low = - covariate_magnitudes[covariate], 
                                                        high = covariate_magnitudes[covariate], 
                                                        size = n_samples)
        else:
            tmp_covariate_by_sample = np.random.uniform(low = - 1, 
                                                        high = 1, 
                                                        size = n_samples)

        # If the current covariate has a beta parameter attached to it 
        if covariate in betas.keys():
            param_base[:, id_tmp] = param_base[:, id_tmp] + (betas[covariate] * tmp_covariate_by_sample)
        else: 
            param_base[:, id_tmp] = param_base[:, id_tmp] + (0.1 * tmp_covariate_by_sample)
    
    #rts = []
    #choices = []

    # TD: IMPROVE THIS SIMULATOR SO THAT WE CAN PASS MATRICES OF PARAMETERS
    # WAY TOO SLOW RIGHT NOW
    #for i in range(n_samples):
    sim_out = simulator(param_base, # param_base[i, :],
                        model = model,
                        n_samples = 1,
                        n_trials = n_samples,
                        bin_dim = bin_dim,
                        bin_pointwise = bin_pointwise,
                        max_t = max_t,
                        delta_t = delta_t)
    
    # sim_out_copy = []
    # sim_out_copy.append(np.squeeze(sim_out[0], axis = 0))
    # sim_out_copy.append(np.squeeze(sim_out[1], axis = 0))
    # sim_out_copy.append(sim_out[2])
        
        #rts.append(sim_out[0])
        #choices.append(sim_out[1])
    
    # rts = np.squeeze(np.stack(rts, axis = 0))
    # choices = np.squeeze(np.stack(choices, axis = 0))
    
    # Preprocess 
    data = hddm_preprocess(sim_out, subj_id)
    # data = hddm_preprocess([rts, choices], subj_id)
    
    # Call the covariate BOLD (unnecessary but in style)
    data['BOLD'] = tmp_covariate_by_sample

    # Make ground truth dict
    gt = {}
    
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        
        # If a parameter actually had a covariate attached then we add the beta coefficient as a parameter as well
        # Now intercept, beta
        if param in betas.keys():
            gt[param + '_beta'] = betas[param]
        
        gt[param] = param_base[0, id_tmp]
    
    return (data, gt)

# ALEX TD: Change n_samples_by_subject --> n_trials_per_subject (but apply consistently)
def simulator_hierarchical(n_subjects = 5,
                           n_samples_by_subject = 500,
                           prespecified_param_means = {'v': 2},
                           prespecified_param_stds = {'v': 0.3},
                           model = 'angle',
                           bin_dim = None,
                           bin_pointwise = True,
                           max_t = 20.0,
                           delta_t = 0.001):

    """Generate a dataset which includes covariates. Some parameters are now a function (regression) covariates.

    :Arguments:
        n_subjects: int <default=5>
            Number of subjects in the datasets
        n_samples_by_subject: int <default=500>
            Number of trials for each subject
        prspecified_param_means: dict <default={'v': 2}>
            Prespeficied group means 
        prespecified_param_stds: dict <default={'v': 0.3}
            Prespeficied group standard deviations
        model: str <default='angle'>
            String that specifies the model to be simulated. 
            Current options include, 'angle', 'ornstein', 'levy', 'weibull', 'full_ddm'
        delta_t: float <default=0.001>
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float <default=20>
            Maximum reaction the simulator can reach
        bin_dim: int <default=None>
            If simulator output should be binned, this specifies the number of bins to use
        bin_pointwise: bool <default=False>
            Determines whether to bin simulator output pointwise. Pointwise here is in contrast to producing binned output
            in the form of a histogram. Binning pointwise gives each trial's RT and index which is the respective bin-number. 
            This is expected when you are using the 'cnn' network to fit the dataset later. If pointwise is not chosen,
            then the takes the form of a histogram, with bin-wise frequencies.

    Returns: 
        (panda.DataFrame, dict, np.array): The Dataframe holds a 'reaction time' column, a 'response' column and a 'BOLD' column (for the covariate). The dictionary holds the groundtruth parameter (values) and parameter names (keys).
                                           Ready to be fit with hddm.
    """

    param_ranges_half = (np.array(model_config[model]['param_bounds'][1]) - np.array(model_config[model]['param_bounds'][0])) / 2
    # Fill in some global parameter vectors
    global_means = np.random.uniform(low = model_config[model]['param_bounds'][0],
                                     high = model_config[model]['param_bounds'][1],
                                     size = (1, len(model_config[model]['param_bounds'][0])))                     

    global_stds = np.random.uniform(low = 0.001, 
                                    high = np.minimum(abs(global_means - model_config[model]['param_bounds'][0]), 
                                                      abs(model_config[model]['param_bounds'][1] - global_means)) / 3, # previously param_ranges_half / 6,
                                    size = (1, len(model_config[model]['param_bounds'][0])))
    
    # global_means = np.random.uniform(low = model_config[model]['param_bounds'][0],
    #                                  high = model_config[model]['param_bounds'][1],
    #                                  size = (1, len(model_config[model]['param_bounds'][0])))                         
    
    dataframes = []
    subject_parameters = np.zeros((n_subjects, 
                                   len(model_config[model]['param_bounds'][0])))
    gt = {}
    
    # Update global parameter vectors according to what was pre-specified
    for param in model_config[model]['params']:
        id_tmp = model_config[model]['params'].index(param)
        
        if param in prespecified_param_means.keys():
            global_means[0, id_tmp] = prespecified_param_means[param]

        if param in prespecified_param_stds.keys():
            global_stds[0, id_tmp] = prespecified_param_means[param]
        
        gt[param] = global_means[0, id_tmp]
        gt[param + '_std'] = global_stds[0, id_tmp]
    
    # For each subject get subject level parameters by sampling from a truncated gaussian as speficied by the global parameters above
    for i in range(n_subjects):
        subj_id = _num_to_str(i)
        
        # Get subject parameters
        a = (model_config[model]['param_bounds'][0] - global_means[0, :]) / global_stds[0, :]
        b = (model_config[model]['param_bounds'][1] - global_means[0, :]) / global_stds[0, :]
        
        subject_parameters[i, :] = np.float32(global_means[0, :] + (truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[0, :]))
        
        sim_out = simulator(subject_parameters[i, :],
                            model = model,
                            n_samples = n_samples_by_subject,
                            bin_dim = bin_dim,
                            bin_pointwise = bin_pointwise,
                            max_t = max_t,
                            delta_t = delta_t)
        
        dataframes.append(hddm_preprocess(simulator_data = sim_out, 
                                          subj_id = subj_id))
        
        for param in model_config[model]['params']:
            id_tmp = model_config[model]['params'].index(param)
            gt[param + '_subj.' + subj_id] = subject_parameters[i, id_tmp]
        
    data_out = pd.concat(dataframes)
    
    return (data_out, gt, subject_parameters)   