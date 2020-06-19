import numpy as np
import keras
from keras.models import Model
from keras.layers import Reshape, ReLU, Input, GlobalAveragePooling2D, multiply, Concatenate, Cropping2D
from keras.layers.core import Activation, Layer
from keras.optimizers import Nadam
from keras import regularizers

DATA_FORMAT = 'channels_last'

def l2_reg():
    return regularizers.l2(2e-5)

def Flatten():
    return keras.layers.Flatten(data_format=DATA_FORMAT)

def Conv2D(filters, kernel_size, use_bias=False, bias_regularizer=None):
    return keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='glorot_normal',
        kernel_regularizer=l2_reg(),
        bias_regularizer=bias_regularizer,
        use_bias=use_bias,
        data_format=DATA_FORMAT)

def BatchNorm(scale):
    return keras.layers.BatchNormalization(
        scale=scale,
        epsilon=1e-5)

def Dense(units, activation, name=None, bias_regularizer=None):
    return keras.layers.Dense(
        units,
        name=name,
        activation=activation,
        kernel_initializer='glorot_normal',
        kernel_regularizer=l2_reg(),
        bias_regularizer=bias_regularizer)

def ConvBlock(filters, kernel_size, batch_scale=False):
    def block(x):
        out = Conv2D(filters=filters, kernel_size=kernel_size)(x)
        out = BatchNorm(scale=batch_scale)(out)
        out = ReLU()(out)

        return out
    return block 

def ResidualBlock(x, filters):
    out = Conv2D(filters=filters, kernel_size=3)(x)
    out = BatchNorm(scale=False)(out)
    out = ReLU()(out)

    out = Conv2D(filters=filters, kernel_size=3)(out)
    out = BatchNorm(scale=True)(out)
    
    out = SqueezeExcitation(out, filters)

    out = keras.layers.add([x, out])
    out = ReLU()(out)

    return out

def SqueezeExcitation(x, filters, ratio=4):
    pool = GlobalAveragePooling2D(data_format=DATA_FORMAT)(x)
    pool = Reshape([1, 1, filters])(pool)
    squeeze = Dense(filters // ratio, activation='relu')(pool)
    excite = Dense(filters, activation='sigmoid')(squeeze)
    return multiply([x, excite])

def ValueHead(x, filters):
    out = ConvBlock(filters=filters // 8, kernel_size=1)(x)

    out = Flatten()(out)
    out = Dense(filters, activation='relu')(out)
    out = Dense(1, name='value_head', activation='tanh')(out)

    return out

def PolicyHeadFullyConnected(x, filters, output_size):
    out = ConvBlock(filters=filters // 8, kernel_size=1)(x)

    out = Flatten()(out)
    out = Dense(output_size, name='policy_head', activation=None, bias_regularizer=l2_reg())(out)
    return out

def PolicyHeadConvolutional(x, filters, output_size):
    conv_block = ConvBlock(filters=filters, kernel_size=3)(x)

    def create_move_dir_out(cropping):
        move_dir_conv = Conv2D(1, kernel_size=3, use_bias=True, bias_regularizer=l2_reg())(conv_block)
        move_dir_cropped = Cropping2D(cropping, data_format=DATA_FORMAT)(move_dir_conv)
        return Flatten()(move_dir_cropped)

    move_up_out = create_move_dir_out(cropping=((1, 0), (0, 0)))
    move_right_out = create_move_dir_out(cropping=((0, 0), (0, 1)))
    move_down_out = create_move_dir_out(cropping=((0, 1), (0, 0)))
    move_left_out = create_move_dir_out(cropping=((0, 0), (1, 0)))

    pass_out = ConvBlock(filters=filters // 8, kernel_size=1)(conv_block)
    pass_out = Flatten()(pass_out)
    pass_out = Dense(filters, activation='relu')(pass_out)
    pass_out = Dense(1, activation=None, bias_regularizer=l2_reg())(pass_out)

    out = Concatenate(name='policy_head')([
        move_up_out,
        move_right_out,
        move_down_out,
        move_left_out,
        pass_out
    ])

    assert out.shape[1], "Policy head size does not match the expected output_size. The convolutional output is currently setup for Arimaa. Either use the PolicyHeadFullyConnected or update this PolicyHeadConvolutional to correspond with your specific game."

    return out

def MovesLeftHead(x, filters, moves_left_size):
    out = ConvBlock(filters=8, kernel_size=1)(x)

    out = Flatten()(out)
    out = Dense(moves_left_size * 2, activation='relu')(out)
    out = Dense(moves_left_size, name='moves_left_head', activation='softmax')(out)
    return out

def create_model(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    inputs = Input(input_shape)
    net = ConvBlock(filters=num_filters, kernel_size=3, batch_scale=True)(inputs)

    for _ in range(0, num_blocks):
        net = ResidualBlock(net, num_filters)

    value_head = ValueHead(net, num_filters)

    # Hack to use the Convolutional head for the Arimaa Play Net
    if output_size == 225:
        policy_head = PolicyHeadConvolutional(net, num_filters, output_size)
    else:
        policy_head = PolicyHeadFullyConnected(net, num_filters, output_size)

    if moves_left_size > 0:
        moves_left_head = MovesLeftHead(net, num_filters, moves_left_size)
        outputs = [value_head, policy_head, moves_left_head]
    else:
        outputs = [value_head, policy_head]

    model = Model(inputs=inputs,outputs=outputs)

    return model
