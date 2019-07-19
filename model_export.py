import c4_model as c4
from keras import backend as K
import tensorflow as tf
from pathlib import Path

def export(model_name):
    import_path = c4.get_model_path(model_name)
    [_, run_name, model_num] = model_name.split('_')[:3]
    model_num = int(model_num)

    K.set_learning_phase(0)

    model = tf.keras.models.load_model(import_path)
    export_path = Path("./exported_models") / run_name / str(model_num)

    print("Inputs", model.input)
    print("Outputs", {t.name: t for t in model.outputs})
    print("Export Path", str(export_path))

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            str(export_path),
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})
