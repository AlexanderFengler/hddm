from tensorflow import keras

def load_mlp(model = 'ddm'):
    if model == 'ddm':
        return keras.models.load_model('model_final_ddm.h5', compile = False)
    else:
        return 'Model is not known'
