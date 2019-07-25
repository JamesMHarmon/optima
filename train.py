import os
import json
from pathlib import Path
from keras import backend as K 
import tensorflow as tf

import c4_model as c4

def export(model_path, export_model_path):
    c4.clear()
    K.set_learning_phase(0)

    model = tf.keras.models.load_model(model_path)

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_model_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})

if __name__== "__main__":

    source_model_path   = os.environ['SOURCE_MODEL_PATH']
    target_model_path   = os.environ['TARGET_MODEL_PATH']
    export_model_path   = os.environ['EXPORT_MODEL_PATH']

    data_path           = os.environ['DATA_PATH']
    train_ratio         = float(os.environ['TRAIN_RATIO'])
    train_batch_size    = int(os.environ['TRAIN_BATCH_SIZE'])
    epochs              = int(os.environ['EPOCHS'])
    learning_rate       = float(os.environ['LEARNING_RATE'])
    policy_loss_weight  = float(os.environ['POLICY_LOSS_WEIGHT'])
    value_loss_weight   = float(os.environ['VALUE_LOSS_WEIGHT'])

    with open(data_path, "r") as read_file:
        data = json.load(read_file)
        X = data["x"]
        yv = data["yv"]
        yp = data["yp"]

    c4.clear()
    model = c4.load_model(source_model_path)

    c4.train(model, X, yv, yp, train_ratio, train_batch_size, epochs, learning_rate, policy_loss_weight, value_loss_weight)

    model.save(target_model_path)

    export(target_model_path, export_model_path)
