from keras.models import load_model
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
import math
import model_sen

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0";

models = {}

def predict(name, p1, p2):
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
    path = get_model_path(name)
    model.save(path)

def get_latest(name):
    directory_path = get_model_directory(name)
    onlyfiles = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    onlynets = [f for f in onlyfiles if f.startswith(name[:-5]) and f.endswith('.h5')]
    onlynets.sort(reverse=True)
    return onlynets[0][:-3]

## PRIVATE...

def get_or_load_model(name):
    path = get_model_path(name)

    if name not in models:
        models[name] = load_model(path)

    return models[name]

# Model name and path convention: ./{game}_runs/{run}/nets/{game}_{run}_{#####}.h5
def get_model_path(name):
    path = get_model_directory(name) + name + '.h5'
    return path

def get_model_directory(name):
    [game, run] = name.split('_')[:2]
    path = './' + game + '_runs/' + run + '/nets/'
    return path


def train(name, from_model_name, target_model_name):
    return