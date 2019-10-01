import os
import json
from pathlib import Path
from keras import backend as K 
import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

import c4_model as c4

def export(model_path, export_model_path):
    c4.clear()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

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

    target_model_path   = os.environ['TARGET_MODEL_PATH']
    export_model_path   = os.environ['EXPORT_MODEL_PATH']

    input_h             = int(os.environ['INPUT_H'])
    input_w             = int(os.environ['INPUT_W'])
    input_c             = int(os.environ['INPUT_C'])
    output_size         = int(os.environ['OUTPUT_SIZE'])
    num_filters         = int(os.environ['NUM_FILTERS'])
    num_blocks          = int(os.environ['NUM_BLOCKS'])

    c4.clear()
    model = c4.create(num_filters, num_blocks, (input_h, input_w, input_c), output_size)

    model.save(target_model_path)

    export(target_model_path, export_model_path)
