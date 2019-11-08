from keras.models import load_model
from keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from pathlib import Path
import os
import numpy as np
import keras
import math
import model_sen

def create(num_filters, num_blocks, input_shape, output_size):
    model = model_sen.create_model(
        num_filters,
        num_blocks,
        input_shape,
        output_size
    )
    
    return model

def train(model, X, yv, yp, train_ratio, train_batch_size, epochs, initial_epoch, learning_rate, policy_loss_weight, value_loss_weight, callbacks):
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
        optimizer=SGD(lr=learning_rate, momentum=0.9),
        loss=loss_funcs,
        loss_weights=loss_weights)

    model.fit(X_train, y_trains,
          batch_size=train_batch_size,
          epochs=epochs,
          initial_epoch=initial_epoch,
          verbose=1,
          validation_data=(X_test, y_tests),
          callbacks=callbacks)

def export(model_path, export_model_path, num_filters, num_blocks, input_shape, output_size):
    model = tf.keras.models.load_model(model_path)
    model_weights = model.get_weights()

    dtype='float16'
    K.clear_session()
    K.set_floatx(dtype)
    K.set_epsilon(1e-4)
    K.set_learning_phase(0)

    model_weights = [w.astype(K.floatx()) for w in model_weights]

    model = create(num_filters, num_blocks, input_shape, output_size)
    model.set_weights(model_weights)

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with K.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_model_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})

def clear():
    K.clear_session()
