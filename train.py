import os
import json
from pathlib import Path
from keras import backend as K 
import tensorflow as tf

import c4_model as c4

# Model name and path convention: ./{game}_runs/{run}/nets/{game}_{run}_{#####}.h5
def get_model_path(name):
    path = get_model_directory(name) / (name + '.h5')
    return path

def get_model_directory(name):
    [game, run] = name.split('_')[:2]
    path = Path(os.path.abspath(os.sep)) / (game + '_runs') / run / 'nets'
    return path

def save_model(name, model):
    directory = get_model_directory(name)
    path = get_model_path(name)

    if not os.path.exists(str(directory)):
        os.makedirs(str(directory))

    model.save(str(path))

def export(model_name):
    c4.clear()
    import_path = get_model_path(model_name)
    [_, run_name, model_num] = model_name.split('_')[:3]
    model_num = int(model_num)

    K.set_learning_phase(0)

    model = tf.keras.models.load_model(str(import_path))
    export_path = Path("./exported_models") / run_name / str(model_num)

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            str(export_path),
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})

if __name__== "__main__":

    source_model_name   = os.environ['SOURCE_MODEL_NAME']
    target_model_name   = os.environ['TARGET_MODEL_NAME']

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
    path = get_model_path(source_model_name)
    model = c4.load_model(str(path))

    c4.train(model, X, yv, yp, train_ratio, train_batch_size, epochs, learning_rate, policy_loss_weight, value_loss_weight)

    save_model(target_model_name, model)

    export(target_model_name)
