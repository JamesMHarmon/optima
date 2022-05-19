from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy as keras_categorical_crossentropy
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
from model_sen import create_model
from policy_head import get_policy_head_fn_by_output_size

def create(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    policy_head = get_policy_head_fn_by_output_size(output_size)

    model = create_model(
        num_filters,
        num_blocks,
        input_shape,
        output_size,
        moves_left_size,
        policy_head=policy_head
    )
    
    return model

def load(model_path):
    return load_model(model_path, custom_objects={'crossentropy_with_policy_mask_loss': crossentropy_with_policy_mask_loss})

def compile(model, learning_rate, policy_loss_weight, value_loss_weight, moves_left_loss_weight):
    loss_funcs = { "value_head": mean_squared_error, "policy_head": crossentropy_with_policy_mask_loss }
    loss_weights = { "value_head": value_loss_weight, "policy_head": policy_loss_weight }

    if any("moves_left" in output.name for output in model.outputs):
        loss_funcs['moves_left_head'] = keras_categorical_crossentropy
        loss_weights['moves_left_head'] = moves_left_loss_weight

    model.compile(
        optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True),
        loss=loss_funcs,
        loss_weights=loss_weights)

def metrics(model):
    accuracy_metrics = { "policy_head": crossentropy_acc }

    if any("moves_left" in output.name for output in model.outputs):
        accuracy_metrics["moves_left_head"] = crossentropy_with_policy_mask_acc

    return accuracy_metrics

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

    tf.saved_model.save(model, export_model_path)

def clear():
    K.clear_session()

def convert_policy_mask(target, predicted):
    is_masked_move = tf.less(target, 0)
    masked_move_logit = tf.zeros_like(predicted) - 1.0e10
    predicted = tf.where(is_masked_move, masked_move_logit, predicted)
    # clamp all targets to be > 0
    target = tf.nn.relu(target)
    return target, predicted

def crossentropy_with_policy_mask_loss(target, predicted):
    target, predicted = convert_policy_mask(target, predicted)
    return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target), logits=predicted)

def crossentropy_with_policy_mask_acc(target, predicted):
    target, predicted = convert_policy_mask(target, predicted)
    return crossentropy_acc(target, predicted)

def crossentropy_acc(target, predicted):
    return tf.cast(tf.equal(tf.argmax(input=target, axis=1), tf.argmax(input=predicted, axis=1)), tf.float32)
