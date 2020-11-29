
# from wfpt import wiener_like_nn_weibull
# from wfpt import wiener_like_nn_angle 
# from wfpt import wiener_like_nn_ddm
# from wfpt import wiener_like_nn_ddm_analytic
# from wfpt import wiener_like_nn_levy
# from wfpt import wiener_like_nn_ornstein
# from wfpt import wiener_like_nn_ddm_sdv
# from wfpt import wiener_like_nn_ddm_sdv_analytic
# from wfpt import wiener_like_nn_full_ddm

import numpy as np
import wfpt

# Defining the likelihood functions
def make_mlp_likelihood(model):
    if model == 'weibull_cdf' or model == 'weibull':
        def wienernn_like_weibull(x, 
                          v,
                          a, 
                          alpha,
                          beta,
                          z,
                          t,
                          p_outlier = 0,
                          w_outlier = 0,
                          **kwargs): #theta

            return wfpt.wiener_like_nn_weibull(x['rt'].values,
                                        x['response'].values, 
                                        v, 
                                        a, 
                                        alpha, 
                                        beta,
                                        z, 
                                        t, 
                                        p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                        w_outlier = w_outlier,
                                        **kwargs)
        return wienernn_like_weibull

    if model == 'ddm':
        def wienernn_like_ddm(x, 
                              v,  
                              a, 
                              z,  
                              t, 
                              p_outlier = 0,
                              w_outlier = 0.1,
                              **kwargs):

            return wfpt.wiener_like_nn_ddm(x['rt'].values,
                                    x['response'].values,  
                                    v, # sv,
                                    a, 
                                    z, # sz,
                                    t, # st,
                                    p_outlier = p_outlier,
                                    w_outlier = w_outlier,
                                    **kwargs)
        return wienernn_like_ddm

    if model == 'ddm_sdv':
        def wienernn_like_ddm_analytic(x, 
                                       v, 
                                       a, 
                                       z, 
                                       t,
                                       p_outlier = 0,
                                       w_outlier = 0,
                                       **kwargs):

            return wfpt.wiener_like_nn_ddm_analytic(x['rt'].values,
                                            x['response'].values,  
                                            v,
                                            a,
                                            z,
                                            t, 
                                            p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                            w_outlier = w_outlier,
                                            **kwargs)
        return wienernn_like_ddm_analytic

    if model == 'ddm_sdv':
        def wienernn_like_ddm_sdv(x, 
                          v,
                          sv,
                          a, 
                          z, 
                          t,
                          p_outlier = 0,
                          w_outlier = 0,
                          **kwargs):

            return wfpt.wiener_like_nn_ddm_sdv(x['rt'].values,
                                        x['response'].values,  
                                        v,
                                        sv,
                                        a,
                                        z, 
                                        t, 
                                        p_outlier = p_outlier,
                                        w_outlier = w_outlier,
                                        **kwargs)
        return wienernn_like_ddm_sdv

    if model == 'ddm_sdv_analytic':
        def wienernn_like_ddm_sdv_analytic(x, 
                                        v, 
                                        sv,
                                        a, 
                                        z, 
                                        t, 
                                        p_outlier = 0,
                                        w_outlier = 0,
                                        **kwargs):

            return wfpt.wiener_like_nn_ddm_sdv_analytic(x['rt'].values,
                                                x['response'].values,  
                                                v, 
                                                sv, 
                                                a, 
                                                z, 
                                                t,
                                                p_outlier = p_outlier,
                                                w_outlier = w_outlier,
                                                **kwargs)
        return wienernn_like_ddm_sdv_analytic

    if model == 'levy':
        def wienernn_like_levy(x, 
                       v, 
                       a, 
                       alpha,
                       z,
                       t,
                       p_outlier = 0.1,
                       w_outlier = 0.1,
                       **kwargs): #theta

            return wfpt.wiener_like_nn_levy(x['rt'].values,
                                    x['response'].values, 
                                    v,
                                    a, 
                                    alpha, 
                                    z,
                                    t, 
                                    p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                    w_outlier = w_outlier,
                                    **kwargs)
        return wienernn_like_levy

    if model == 'ornstein':
        def wienernn_like_ornstein(x, 
                           v, 
                           a, 
                           g,
                           z, 
                           t,
                           p_outlier = 0,
                           w_outlier = 0,
                           **kwargs): #theta
    
            return wfpt.wiener_like_nn_ornstein(x['rt'].values,
                                        x['response'].values, 
                                        v, 
                                        a, 
                                        g, 
                                        z, 
                                        t, 
                                        p_outlier = p_outlier, # TODO: ACTUALLY USE THIS
                                        w_outlier = w_outlier,
                                        **kwargs)
        return wienernn_like_ornstein

    if model == 'full_ddm' or model == 'full_ddm2':
        def wienernn_like_full_ddm(x, 
                           v, 
                           sv, 
                           a, 
                           z, 
                           sz, 
                           t, 
                           st, 
                           p_outlier = 0,
                           w_outlier = 0,
                           **kwargs):

            return wfpt.wiener_like_nn_full_ddm(x['rt'].values,
                                        x['response'].values,  
                                        v, 
                                        sv, 
                                        a, 
                                        z, 
                                        sz, 
                                        t, 
                                        st, 
                                        p_outlier = p_outlier,
                                        w_outlier = w_outlier,
                                        **kwargs)

        return wienern_like_full_ddm

    if model == 'angle':
        def wienernn_like_angle(x, 
                        v, 
                        a, 
                        theta, 
                        z,
                        t,
                        p_outlier = 0,
                        w_outlier = 0,
                        **kwargs):

            return wfpt.wiener_like_nn_angle(x['rt'].values,
                                        x['response'].values,  
                                        v,
                                        a, 
                                        theta,
                                        z,
                                        t,
                                        p_outlier = p_outlier,
                                        w_outlier = w_outlier,
                                        **kwargs)
        
        return wienernn_like_angle
    else:
        return 'Not implemented errror: Failed to load likelihood because the model specified is not implemented'


