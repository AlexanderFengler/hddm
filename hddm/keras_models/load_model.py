from tensorflow import keras
import hddm.keras_models

def load_mlp(model = 'ddm'):
    if model == 'ddm':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ddm.h5', compile = False)
    if model == 'ddm_analytic':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ddm_analytic.h5', compile = False)
    
    if model == 'weibull_cdf':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_new.h5', compile = False)
    
    if model == 'angle':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_angle.h5', compile = False)
    
    if model == 'ornstein':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ornstein.h5', compile = False)

    if model == 'levy':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_levy.h5', compile = False)
    
    if model == 'full_ddm' or model == 'full_ddm2':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_full_ddm.h5', compile = False)
    
    if model == 'ddm_sdv':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ddm_sdv.h5', compile = False)
    
    if model == 'ddm_sdv_analytic':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ddm_sdv_analytic.h5', compile = False)

    if model == 'ddm_sdv_analytic:':
        return keras.models.load_model(hddm.keras_models.__path__ + '/model_final_ddm_sdv_analytic.h5', compile = False)
    
    else:
        return 'Model is not known'


# weibull_model = keras.models.load_model('model_final_weibull.h5', compile = False)
# angle_model = keras.models.load_model('model_final_angle.h5', compile = False)
# #model = keras.models.load_model('model_final.h5', compile = False)
# new_weibull_model = keras.models.load_model('model_final_new.h5', compile = False)
# ddm_model = keras.models.load_model('model_final_ddm.h5', compile = False)
# ddm_analytic_model = keras.models.load_model('model_final_ddm_analytic.h5', compile = False)
# levy_model = keras.models.load_model('model_final_levy.h5', compile = False)
# ornstein_model = keras.models.load_model('model_final_ornstein.h5', compile = False)
# ddm_sdv_model = keras.models.load_model('model_final_ddm_sdv.h5', compile = False)
# ddm_sdv_analytic_model = keras.models.load_model('model_final_ddm_sdv_analytic.h5', compile = False)
# full_ddm_model = keras.models.load_model('model_final_full_ddm.h5', compile = False)
