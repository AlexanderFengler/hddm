# cython: embedsignature=True
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# distutils: language = c++
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm

import scipy.integrate as integrate
from copy import copy
import numpy as np

#for the nn_like
#import ckeras_to_numpy as ktnp
from tensorflow import keras
import pandas as pd

cimport numpy as np
cimport cython

from cython.parallel import *
from hddm.keras_models import load_mlp

# cimport openmp

# include "pdf.pxi"
include 'integrate.pxi'


# LOADING MODELS
# TODO: Refactor this ?
#ddm_model = load_mlp(model = 'ddm')

weibull_model = load_mlp(model == 'weibull_cdf') # keras.models.load_model('model_final_weibull.h5', compile = False)
angle_model = load_mlp(model == 'angle') #keras.models.load_model('model_final_angle.h5', compile = False)
#model = keras.models.load_model('model_final.h5', compile = False)
new_weibull_model = load_mlp(model == 'weibull_cdf') # keras.models.load_model('model_final_new.h5', compile = False)
ddm_model = load_mlp(model == 'ddm') # keras.models.load_model('model_final_ddm.h5', compile = False)
ddm_analytic_model = load_mlp(model = 'ddm_analytic') #keras.models.load_model('model_final_ddm_analytic.h5', compile = False)
levy_model = load_mlp(model == 'levy') # keras.models.load_model('model_final_levy.h5', compile = False)
ornstein_model = load_mlp(model == 'ornstein') # keras.models.load_model('model_final_ornstein.h5', compile = False)
ddm_sdv_model = load_mlp(model == 'ddm_sdv') # keras.models.load_model('model_final_ddm_sdv.h5', compile = False)
ddm_sdv_analytic_model = load_mlp(model == 'ddm_sdv_analytic') # keras.models.load_model('model_final_ddm_sdv_analytic.h5', compile = False)
full_ddm_model = load_mlp(model = 'full_ddm') #keras.models.load_model('model_final_full_ddm.h5', compile = False)

###############
# Basic Navarro Fuss likelihoods
def pdf_array(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz,
              double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1,
              double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim = 1] y = np.empty(size, dtype=np.double)

    for i in prange(size, nogil=True):
        y[i] = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                        n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp == 1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier):
    return (p_outlier >= 0) & (p_outlier <= 1)


def wiener_like(np.ndarray[double, ndim=1] x, double v, double sv, double a, double z, double sz, double t,
                double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    for i in range(size):
        p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                     n_st, n_sz, use_adaptive, simps_err)
        # If one probability = 0, the log sum will be -Inf
        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf

        sum_logp += log(p)

    return sum_logp

###########
# Basic MLP likelihoods

def wiener_like_nn_full_ddm(np.ndarray[float, ndim = 1] x, 
                            np.ndarray[float, ndim = 1] response, 
                            double v, 
                            double sv, 
                            double a, 
                            double z, 
                            double sz, 
                            double t, 
                            double st, 
                            double p_outlier = 0, 
                            double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 7
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t, sz, sv, st], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(full_ddm_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(full_ddm_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

def wiener_like_nn_ddm(np.ndarray[float, ndim = 1] x, 
                       np.ndarray[float, ndim = 1] response, 
                       double v, # double sv,
                       double a, 
                       double z, # double sz,
                       double t, #  double st,
                       double p_outlier = 0, 
                       double w_outlier = 0):

    
    # double err = 1e-4, 
    # int n_st = 10, 
    # int n_sz = 10, 
    # bint use_adaptive = 1,
    # double simps_err = 1e-8,

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 4
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    #print(p_outlier)
    #if not p_outlier_in_range(p_outlier):
    #    return -np.inf
    
    # Call to network:
    if p_outlier == 0: # ddm_model
        log_p = np.sum(np.core.umath.maximum(mlp.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(mlp.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
        #log_p = np.sum(np.log(np.add(np.multiply(np.exp(np.core.umath.maximum(ddm_model.predict_on_batch(data), ll_min)), 
        #                    (1.0 - p_outlier)), 
        #              (w_outlier * p_outlier))))
    return log_p

    

def wiener_like_nn_ddm_analytic(np.ndarray[float, ndim = 1] x, 
                                np.ndarray[float, ndim = 1] response, 
                                double v, 
                                double a, 
                                double z, 
                                double t, 
                                double p_outlier = 0, 
                                double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 4
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ddm_analytic_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ddm_analytic_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    return log_p

def wiener_like_nn_angle(np.ndarray[float, ndim = 1] x, 
                         np.ndarray[float, ndim = 1] response, 
                         double v,
                         double a, 
                         double theta, 
                         double z,
                         double t,
                         double p_outlier=0, 
                         double w_outlier=0):
    
    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 5
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t, theta], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(angle_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(angle_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    
    return log_p

def wiener_like_nn_weibull(np.ndarray[float, ndim = 1] x, 
                           np.ndarray[float, ndim = 1] response, 
                           double v,
                           double a, 
                           double alpha, 
                           double beta, 
                           double z, 
                           double t,
                           double p_outlier = 0,
                           double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 6
    cdef float ll_min = -16.11809
    
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t, alpha, beta], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(new_weibull_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(new_weibull_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    
    return log_p
#
def wiener_like_nn_levy(np.ndarray[float, ndim = 1] x,
                        np.ndarray[float, ndim = 1] response, 
                        double v,
                        double a, 
                        double alpha,
                        double z, 
                        double t,
                        double p_outlier = 0,
                        double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 5
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, alpha, t], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(levy_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(levy_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    
    return log_p

def wiener_like_nn_ornstein(np.ndarray[float, ndim = 1] x, 
                            np.ndarray[float, ndim = 1] response, 
                            double v,
                            double a, 
                            double g,
                            double z, 
                            double t, 
                            double p_outlier = 0,
                            double w_outlier = 0):
    
    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 5
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, g, t], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ornstein_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ornstein_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))
    
    return log_p

def wiener_like_nn_ddm_sdv(np.ndarray[float, ndim = 1] x, 
                           np.ndarray[float, ndim = 1] response, 
                           double v,
                           double sv,
                           double a,
                           double z,
                           double t,
                           double p_outlier = 0,
                           double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 5
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t, sv], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ddm_sdv_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ddm_sdv_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

def wiener_like_nn_ddm_sdv_analytic(np.ndarray[float, ndim = 1] x, 
                                    np.ndarray[float, ndim = 1] response, 
                                    double v,
                                    double sv, 
                                    double a,
                                    double z, 
                                    double t,
                                    double p_outlier = 0,
                                    double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef float log_p
    cdef int n_params = 5
    cdef float ll_min = -16.11809
    cdef np.ndarray[float, ndim = 2] data = np.zeros((size, n_params + 2), dtype = np.float32)
    data[:, :n_params] = np.tile([v, a, z, t, sv], (size, 1)).astype(np.float32)
    data[:, n_params:] = np.stack([x, response], axis = 1)

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ddm_sdv_analytic_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ddm_sdv_analytic_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p

###############
# Regression style likelihoods: (Can prob simplify and make all mlp likelihoods of this form)

def wiener_like_multi_nn_ddm(np.ndarray[float, ndim = 2] data,
                             double p_outlier = 0, 
                             double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ddm_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ddm_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    #log_p = np.sum(np.core.umath.maximum(ddm_model.predict_on_batch(data), ll_min))
    return log_p 

def wiener_like_multi_nn_angle(np.ndarray[float, ndim = 2] data,
                               double p_outlier = 0, 
                               double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(angle_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(angle_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    #log_p = np.sum(np.core.umath.maximum(angle_model.predict_on_batch(data), ll_min))
    return log_p 

def wiener_like_multi_nn_weibull(np.ndarray[float, ndim = 2] data,
                                 double p_outlier = 0, 
                                 double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p
    
    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(weibull_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(weibull_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    #log_p = np.sum(np.core.umath.maximum(new_weibull_model.predict_on_batch(data), ll_min))
    return log_p 

def wiener_like_multi_nn_levy(np.ndarray[float, ndim = 2] data,
                              double p_outlier = 0, 
                              double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    if (np.min(data[:, 3]) < 1.0):
        return - np.inf
    if (np.max(data[:, 3] > 2.0)):
        return - np.inf

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(levy_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(levy_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    return log_p 

def wiener_like_multi_nn_ornstein(np.ndarray[float, ndim = 2] data,
                                  double p_outlier = 0, 
                                  double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    # Call to network:
    if p_outlier == 0:
        log_p = np.sum(np.core.umath.maximum(ornstein_model.predict_on_batch(data), ll_min))
    else:
        log_p = np.sum(np.log(np.exp(np.core.umath.maximum(ornstein_model.predict_on_batch(data), ll_min)) * (1.0 - p_outlier) + (w_outlier * p_outlier)))

    # log_p = np.sum(np.core.umath.maximum(ornstein_model.predict_on_batch(data), ll_min))
    return log_p 

def wiener_like_multi_nn_full_ddm(np.ndarray[float, ndim = 2] data,
                                  double p_outlier = 0, 
                                  double w_outlier = 0):
    
    cdef float ll_min = -16.11809
    cdef float log_p

    log_p = np.sum(np.core.umath.maximum(full_ddm_model.predict_on_batch(data), ll_min))
    return log_p 

# RL - DDM
def wiener_like_rlddm(np.ndarray[double, ndim=1] x,
                      np.ndarray[long, ndim=1] response,
                      np.ndarray[double, ndim=1] feedback,
                      np.ndarray[long, ndim=1] split_by,
                      double q, double alpha, double pos_alpha, double v, 
                      double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] xs
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha

    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        xs = x[split_by == s]
        s_size = xs.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):
            p = full_pdf(xs[i], ((qs[1] - qs[0]) * v), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf
            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp

def wiener_like_rl(np.ndarray[long, ndim=1] response,
                   np.ndarray[double, ndim=1] feedback,
                   np.ndarray[long, ndim=1] split_by,
                   double q, double alpha, double pos_alpha, double v, double z,
                   double err=1e-4, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                   double p_outlier=0, double w_outlier=0):
    cdef Py_ssize_t size = response.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t s_size
    cdef int s
    cdef double drift
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double pos_alfa
    cdef np.ndarray[double, ndim=1] qs = np.array([q, q])
    cdef np.ndarray[double, ndim=1] feedbacks
    cdef np.ndarray[long, ndim=1] responses
    cdef np.ndarray[long, ndim=1] unique = np.unique(split_by)

    if not p_outlier_in_range(p_outlier):
        return -np.inf

    if pos_alpha==100.00:
        pos_alfa = alpha
    else:
        pos_alfa = pos_alpha
        
    # unique represent # of conditions
    for j in range(unique.shape[0]):
        s = unique[j]
        # select trials for current condition, identified by the split_by-array
        feedbacks = feedback[split_by == s]
        responses = response[split_by == s]
        s_size = responses.shape[0]
        qs[0] = q
        qs[1] = q

        # don't calculate pdf for first trial but still update q
        if feedbacks[0] > qs[responses[0]]:
            alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
        else:
            alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

        # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
        # received on current trial.
        qs[responses[0]] = qs[responses[0]] + \
            alfa * (feedbacks[0] - qs[responses[0]])

        # loop through all trials in current condition
        for i in range(1, s_size):

            drift = (qs[1] - qs[0]) * v

            if drift == 0:
                p = 0.5
            else:
                if responses[i] == 1:
                    p = (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)
                else:
                    p = 1 - (2.718281828459**(-2 * z * drift) - 1) / \
                        (2.718281828459**(-2 * drift) - 1)

            # If one probability = 0, the log sum will be -Inf
            p = p * (1 - p_outlier) + wp_outlier
            if p == 0:
                return -np.inf

            sum_logp += log(p)

            # get learning rate for current trial. if pos_alpha is not in
            # include it will be same as alpha so can still use this
            # calculation:
            if feedbacks[i] > qs[responses[i]]:
                alfa = (2.718281828459**pos_alfa) / (1 + 2.718281828459**pos_alfa)
            else:
                alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)

            # qs[1] is upper bound, qs[0] is lower bound. feedbacks is reward
            # received on current trial.
            qs[responses[i]] = qs[responses[i]] + \
                alfa * (feedbacks[i] - qs[responses[i]])
    return sum_logp


def wiener_like_multi(np.ndarray[double, ndim = 1] x, v, sv, a, z, sz, t, st, 
                      double err, 
                      multi = None,
                      int n_st = 10, 
                      int n_sz = 10, 
                      bint use_adaptive = 1, 
                      double simps_err = 1e-3,
                      double p_outlier = 0, 
                      double w_outlier = 0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p = 0
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier

    if multi is None:
        return full_pdf(x, v, sv, a, z, sz, t, st, err)
    else:
        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st}
        params_iter = copy(params)
        for i in range(size):
            for param in multi:
                params_iter[param] = params[param][i]

            p = full_pdf(x[i], params_iter['v'],
                         params_iter['sv'], params_iter['a'], params_iter['z'],
                         params_iter['sz'], params_iter['t'], params_iter['st'],
                         err, n_st, n_sz, use_adaptive, simps_err)
            p = p * (1 - p_outlier) + wp_outlier
            sum_logp += log(p)

        return sum_logp

def gen_rts_from_cdf(double v, double sv, double a, double z, double sz, double t,
                     double st, int samples=1000, double cdf_lb=-6, double cdf_ub=6, double dt=1e-2):

    cdef np.ndarray[double, ndim = 1] x = np.arange(cdf_lb, cdf_ub, dt)
    cdef np.ndarray[double, ndim = 1] l_cdf = np.empty(x.shape[0], dtype=np.double)
    cdef double pdf, rt
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i, j
    cdef int idx

    l_cdf[0] = 0
    for i from 1 <= i < size:
        pdf = full_pdf(x[i], v, sv, a, z, sz, 0, 0, 1e-4)
        l_cdf[i] = l_cdf[i - 1] + pdf

    l_cdf /= l_cdf[x.shape[0] - 1]

    cdef np.ndarray[double, ndim = 1] rts = np.empty(samples, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] f = np.random.rand(samples)
    cdef np.ndarray[double, ndim = 1] delay

    if st != 0:
        delay = (np.random.rand(samples) * st + (t - st / 2.))
    for i from 0 <= i < samples:
        idx = np.searchsorted(l_cdf, f[i])
        rt = x[idx]
        if st == 0:
            rt = rt + np.sign(rt) * t
        else:
            rt = rt + np.sign(rt) * delay[i]
        rts[i] = rt
    return rts


def wiener_like_contaminant(np.ndarray[double, ndim=1] x, np.ndarray[int, ndim=1] cont_x, double v,
                            double sv, double a, double z, double sz, double t, double st, double t_min,
                            double t_max, double err, int n_st=10, int n_sz=10, bint use_adaptive=1,
                            double simps_err=1e-8):
    """Wiener likelihood function where RTs could come from a
    separate, uniform contaminant distribution.

    Reference: Lee, Vandekerckhove, Navarro, & Tuernlinckx (2007)
    """
    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0
    cdef int n_cont = np.sum(cont_x)
    cdef int pos_cont = 0

    for i in prange(size, nogil=True):
        if cont_x[i] == 0:
            p = full_pdf(x[i], v, sv, a, z, sz, t, st, err,
                         n_st, n_sz, use_adaptive, simps_err)
            if p == 0:
                with gil:
                    return -np.inf
            sum_logp += log(p)
        # If one probability = 0, the log sum will be -Inf

    # add the log likelihood of the contaminations
    sum_logp += n_cont * log(0.5 * 1. / (t_max - t_min))

    return sum_logp


def gen_cdf_using_pdf(double v, double sv, double a, double z, double sz, double t, double st, double err,
                      int N=500, double time=5., int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3,
                      double p_outlier=0, double w_outlier=0):
    """
    generate cdf vector using the pdf
    """
    if (sv < 0) or (a <= 0 ) or (z < 0) or (z > 1) or (sz < 0) or (sz > 1) or (z + sz / 2. > 1) or \
            (z - sz / 2. < 0) or (t - st / 2. < 0) or (t < 0) or (st < 0) or not p_outlier_in_range(p_outlier):
        raise ValueError(
            "at least one of the parameters is out of the support")

    cdef np.ndarray[double, ndim = 1] x = np.linspace(-time, time, 2 * N + 1)
    cdef np.ndarray[double, ndim = 1] cdf_array = np.empty(x.shape[0], dtype=np.double)
    cdef int idx

    # compute pdf on the real line
    cdf_array = pdf_array(x, v, sv, a, z, sz, t, st, err, 0,
                          n_st, n_sz, use_adaptive, simps_err, p_outlier, w_outlier)

    # integrate
    cdf_array[1:] = integrate.cumtrapz(cdf_array)

    # normalize
    cdf_array /= cdf_array[x.shape[0] - 1]

    return x, cdf_array


def split_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] data):

    # get length of data
    cdef int N = (len(data) - 1) / 2

    # lower bound is reversed
    cdef np.ndarray[double, ndim = 1] x_lb = -x[:N][::-1]
    cdef np.ndarray[double, ndim = 1] lb = data[:N][::-1]
    # lower bound is cumulative in the wrong direction
    lb = np.cumsum(np.concatenate([np.array([0]), -np.diff(lb)]))

    cdef np.ndarray[double, ndim = 1] x_ub = x[N + 1:]
    cdef np.ndarray[double, ndim = 1] ub = data[N + 1:]
    # ub does not start at 0
    ub -= ub[0]

    return (x_lb, lb, x_ub, ub)


###### 
# UNUSED


#def wiener_like_multi_nnddm(np.ndarray[double, ndim=1] x, 
#                            np.ndarray[long, ndim=1] response, 
#                            activations, 
#                            weights, 
#                            biases, 
#                            v, 
#                            sv, 
#                            a, 
#                            z, 
#                            sz, 
#                            t, 
#                            st, 
#                            alpha, 
#                            beta, 
#                            double err, 
#                            multi=None,
#                            int n_st=10, 
#                            int n_sz=10, 
#                            bint use_adaptive=1, 
#                            double simps_err=1e-3,
#                            double p_outlier=0, 
#                            double w_outlier=0):
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef double p = 0
#    cdef double sum_logp = 0
#    cdef double wp_outlier = w_outlier * p_outlier

#    if multi is None:
#        return full_pdf(x, v, sv, a, z, sz, t, st, err)
#    else:
#        params = {'v': v, 'z': z, 't': t, 'a': a, 'sv': sv, 'sz': sz, 'st': st, 'alpha':alpha, 'beta': beta}
#        params_iter = copy(params)
#        for i in range(size):
#            for param in multi:
#                params_iter[param] = params[param][i]

            #print(type(params_iter['a'].astype(float)))
            #print(params_iter['v'][0])
            #print(params_iter['z'])
            #print(params_iter['t'])
            #print(params_iter['theta'])
            #print(params_iter['a'])
            #print(i)
            #print(x[i])


 #           p = 0.1 #ktnp.predict(np.array([params_iter['v'][0],params_iter['a'],params_iter['z'],params_iter['t'],params_iter['alpha'],params_iter['beta'],x[i], response[i]]), weights, biases, activations, len(activations))

            #print(p)
            #full_pdf(x[i], params_iter['v'],
            #             params_iter['sv'], params_iter['a'], params_iter['z'],
            #             params_iter['sz'], params_iter[
            #                 't'], params_iter['st'],
            #             err, n_st, n_sz, use_adaptive, simps_err)
 #           p = p * (1 - p_outlier) + wp_outlier
 #           sum_logp += p
#
# #       return sum_logp
#
#def wiener_like_nn_weibull(np.ndarray[double, ndim = 1] x, 
#                           np.ndarray[long, ndim = 1] response, 
#                           double v,
#                           double sv, 
#                           double a, 
#                           double alpha, 
#                           double beta, 
#                           double z, 
#                           double sz, 
#                           double t,
#                           double st, 
#                           double err, 
#                           int n_st=10, 
#                           int n_sz=10, 
#                           bint use_adaptive = 1, 
#                           double simps_err = 1e-8,
#                           double p_outlier = 0,
#                           double w_outlier = 0):
#
#    cdef Py_ssize_t size = x.shape[0]
#    cdef Py_ssize_t i
#    cdef double p
#    cdef double sum_logp = 0
#    cdef double wp_outlier = w_outlier * p_outlier
#    cdef double n_params = 6
#
#    cdef np.ndarray[double, ndim=1] vf = np.repeat(v,size)
#    cdef np.ndarray[double, ndim=1] af = np.repeat(a,size)
#    cdef np.ndarray[double, ndim=1] zf = np.repeat(z,size)
#    cdef np.ndarray[double, ndim=1] tf = np.repeat(t,size)
#    cdef np.ndarray[double, ndim=1] betaf = np.repeat(beta,size)
#    cdef np.ndarray[double, ndim=1] alphaf = np.repeat(alpha,size)
#
#    if not p_outlier_in_range(p_outlier):
#        return -np.inf
#
#    p = mlp_target_weibull(np.array([vf,af,zf,tf,alphaf,betaf]).transpose(),np.array([x,response]).transpose())
#
#    if p == 0:
#        return -np.inf
#
#    return p
#

#def mlp_target_weibull(np.ndarray[double, ndim = 2] params,
#                       np.ndarray[double, ndim = 2] data, 
#                       ll_min = -16.11809 # corresponds to 1e-7
#                       ): 
#
#    n_params = 6
#    mlp_input_batch = np.zeros((data.shape[0], params.shape[1] + 2), dtype = np.float32)
#    mlp_input_batch[:, :n_params] = params
#    mlp_input_batch[:, n_params:] = data
#    #return np.sum(np.core.umath.maximum(ktnp.predict(mlp_input_batch, weights, biases, activations, n_layers), ll_min))
#    return np.sum(np.core.umath.maximum(weibull_model.predict_on_batch(mlp_input_batch), ll_min))
