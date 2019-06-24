from keras.models import load_model
import numpy as np
import keras
import math
import model_sen

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0";

model = None

def convertGameStateToInput(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1_board = p1.reshape(6, 7)
    p2_board = p2.reshape(6, 7)
    return np.stack((p1_board, p2_board), axis=2)

def analyse(p1, p2):
    global model

    input = convertGameStateToInput(p1, p2)

    prediction = model.predict(np.asarray([input]))
    value = prediction[0][0][0]
    policy = prediction[1][0]

    return (value, policy)

def create_model():
    global model

    model = model_sen.compile_model(
        num_filters=64,
        num_blocks=5,
        input_shape=(6, 7, 2)
    )
    return

def load_model_(path):
    global model
    model = load_model(path)
    return model

def train_model():
    return