import os
import json
from pathlib import Path
from keras import backend as K 
import tensorflow as tf
from keras.callbacks import TensorBoard

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
    tensor_board_path   = os.environ['TENSOR_BOARD_PATH']

    data_paths          = os.environ['DATA_PATHS'].split(',')
    train_ratio         = float(os.environ['TRAIN_RATIO'])
    train_batch_size    = int(os.environ['TRAIN_BATCH_SIZE'])
    epochs              = int(os.environ['EPOCHS'])
    initial_epoch       = int(os.environ['INITIAL_EPOCH'])
    learning_rate       = float(os.environ['LEARNING_RATE'])
    policy_loss_weight  = float(os.environ['POLICY_LOSS_WEIGHT'])
    value_loss_weight   = float(os.environ['VALUE_LOSS_WEIGHT'])

    print(data_paths)
    c4.clear()
    
    for i, path in enumerate(data_paths):
        with open(path, "r") as read_file:
            data = json.load(read_file)
            X = data["x"]
            yv = data["yv"]
            yp = data["yp"]

        model = c4.load_model(source_model_path)

        tensor_board = TensorBoard(log_dir=tensor_board_path,update_freq='epoch')
        callbacks = [tensor_board] if i == 0 else []

        c4.train(model, X, yv, yp, train_ratio, train_batch_size, epochs, initial_epoch, learning_rate, policy_loss_weight, value_loss_weight, callbacks)

    model.save(target_model_path)

    export(target_model_path, export_model_path)
