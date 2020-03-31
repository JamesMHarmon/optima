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
import warmup_lr_scheduler

def create(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    model = model_sen.create_model(
        num_filters,
        num_blocks,
        input_shape,
        output_size,
        moves_left_size
    )
    
    return model

def load(model_path):
    return load_model(model_path, custom_objects={'categorical_crossentropy_from_logits': categorical_crossentropy_from_logits})

def train(model, X, yv, yp, ym, train_ratio, train_batch_size, epochs, initial_epoch, max_grad_norm, learning_rate, policy_loss_weight, value_loss_weight, moves_left_loss_weight, callbacks):
    X_train, X_test, yv_train, yv_test, yp_train, yp_test, ym_train, ym_test = train_test_split(
        X,
        yv,
        yp,
        ym,
        train_size=train_ratio,
        shuffle=False)

    X_train, yv_train, yp_train, ym_train = clip_to_be_divisible(X_train, yv_train, yp_train, ym_train, divisor=train_batch_size)

    if len(X_train) == 0:
        return

    y_trains = { "value_head": yv_train, "policy_head": yp_train }
    y_tests = { "value_head": yv_test, "policy_head": yp_test }
    loss_funcs = { "value_head": "mean_squared_error", "policy_head": categorical_crossentropy_from_logits }
    loss_weights = { "value_head": value_loss_weight, "policy_head": policy_loss_weight }

    if any("moves_left" in output.name for output in model.outputs):
        y_trains['moves_left_head'] = ym_train
        y_tests['moves_left_head'] = ym_test
        loss_funcs['moves_left_head'] = "categorical_crossentropy"
        loss_weights['moves_left_head'] = moves_left_loss_weight

    steps_per_epoch = X_train.shape[0] // train_batch_size
    lr_schedule = warmup_lr_scheduler.WarmupLearningRateScheduler(lr=learning_rate, warmup_steps=1000, steps_per_epoch=steps_per_epoch)

    model.compile(
        optimizer=SGD(lr=learning_rate, momentum=0.9, clipnorm=max_grad_norm),
        loss=loss_funcs,
        loss_weights=loss_weights)

    model.fit(X_train, y_trains,
        batch_size=train_batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        validation_data=(X_test, y_tests),
        callbacks=callbacks + [lr_schedule])

def export(model_path, export_model_path, num_filters, num_blocks, input_shape, output_size, moves_left_size):
    model = load(model_path)
    model_weights = model.get_weights()

    dtype='float16'
    K.clear_session()
    K.set_floatx(dtype)
    K.set_epsilon(1e-4)
    K.set_learning_phase(0)

    model_weights = [w.astype(K.floatx()) for w in model_weights]

    model = create(num_filters, num_blocks, input_shape, output_size, moves_left_size)
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

def categorical_crossentropy_from_logits(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0)

def clip_to_be_divisible(*args, divisor):
    size = args[0].shape[0]
    clipped_size = (size // divisor) * divisor

    result = []
    for arg in args:
      result.append(arg[:clipped_size])

    return tuple(result)
