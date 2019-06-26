from keras.models import load_model
import numpy as np
import keras
import math
import model_sen

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0";

models = {}

def analyse(name, p1, p2):
    model = get_or_load_model(name)
    input = convertGameStateToInput(p1, p2)

    prediction = model.predict(np.asarray([input]))
    value = prediction[0][0][0]
    policy = prediction[1][0]

    return (value, policy)

def convertGameStateToInput(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1_board = p1.reshape(6, 7)
    p2_board = p2.reshape(6, 7)
    return np.stack((p1_board, p2_board), axis=2)

def create(name):
    model = model_sen.compile_model(
        num_filters=64,
        num_blocks=5,
        input_shape=(6, 7, 2)
    )

    models[name] = model
    model.save(get_model_path(name))

def get_or_load_model(name):
    path = get_model_path(name)

    if name not in models:
        models[name] = load_model(path)

    return models[name]

# Model name and path convention: ./{game}/{run}/nets/{game}_{run}_{#####}.h5
def get_model_path(name):
    [game, run] = name.split('_')
    path = './' + game + '/' + run + '/nets/' + name + '.h5'
    return path


def train(name, from_model_name, target_model_name):
    return