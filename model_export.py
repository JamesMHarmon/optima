import c4_model as c4
from keras import backend as K
import tensorflow as tf

def export(model_name):
    import_path = c4.get_model_path(model_name)

    print(import_path)

    K.set_learning_phase(0)

    model = tf.keras.models.load_model(import_path)
    export_path = '.\\export_model_2\\1'

    print("Inputs", model.input)
    print("Outputs", {t.name: t for t in model.outputs})

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})

export("Connect4_run-1_00001")