from keras.models import load_model
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import keras
import math
import model_sen

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0";

models = {}

def predict(name, game_position):
    model = get_or_load_model(name)
    input = convertGameStateToInput(game_position)

    prediction = model.predict(input)
    value = prediction[0][0][0]
    policy = prediction[1][0]

    return (value, policy)

def convertGameStateToInput(game_position):
    return np.asarray(game_position).reshape(1, 6, 7, 2)

def create(name, num_filters, num_blocks, input_shape):
    model = model_sen.create_model(
        num_filters,
        num_blocks,
        input_shape
    )
    
    save_model(name, model)

def get_latest(name):
    directory_path = get_model_directory(name)
    onlyfiles = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    onlynets = [f for f in onlyfiles if f.startswith(name[:-5]) and f.endswith('.h5')]
    onlynets.sort(reverse=True)
    return onlynets[0][:-3]

def train(source_model_name, target_model_name, X, yv, yp, train_ratio, train_batch_size, epochs, learning_rate, policy_loss_weight, value_loss_weight):
    path = get_model_path(source_model_name)
    model = load_model(path)

    X = np.asarray(X)
    yv = np.asarray(yv)
    yp = np.asarray(yp)

    X_train, X_test, yv_train, yv_test, yp_train, yp_test = train_test_split(
        X,
        yv,
        yp,
        train_size=train_ratio)

    y_trains = { "value_head": yv_train, "policy_head": yp_train }
    y_tests = { "value_head": yv_test, "policy_head": yp_test }
    loss_funcs = { "value_head": "mean_squared_error", "policy_head": "categorical_crossentropy" }
    loss_weights = { "value_head": value_loss_weight, "policy_head": policy_loss_weight }

    model.compile(
        optimizer=Nadam(lr=learning_rate),
        loss=loss_funcs,
        loss_weights=loss_weights)

    model.fit(X_train, y_trains,
          batch_size=256,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_tests))

    save_model(target_model_name, model)

## PRIVATE...

def save_model(name, model):
    models[name] = model
    directory = get_model_directory(name)
    path = get_model_path(name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model.save(path)

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
