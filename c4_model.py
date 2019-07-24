from keras.models import load_model
from keras.optimizers import Nadam
from keras import backend as K 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from pathlib import Path
import os
import numpy as np
import keras
import math
import model_sen

def create(name, num_filters, num_blocks, input_shape):
    model = model_sen.create_model(
        num_filters,
        num_blocks,
        input_shape
    )
    
    save_model(name, model)

def train(model, X, yv, yp, train_ratio, train_batch_size, epochs, learning_rate, policy_loss_weight, value_loss_weight):
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
          batch_size=train_batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_tests))

## PRIVATE...

def clear():
    K.clear_session()
