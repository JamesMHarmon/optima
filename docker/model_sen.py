import numpy as np
import keras
from keras.models import Model
from keras.layers import Reshape, Flatten, ReLU, Input, GlobalAveragePooling2D, multiply
from keras.layers.core import Activation, Layer
from keras.optimizers import Nadam
from keras import regularizers

def l2_reg():
    return regularizers.l2(0.00005)

def Conv2D(filters, kernel_size):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2_reg(), use_bias=False)

def BatchNorm(scale):
    gamma_regularizer = None
    if scale:
        gamma_regularizer=l2_reg()

    return keras.layers.BatchNormalization(scale=scale, epsilon=1e-5, beta_regularizer=l2_reg(), gamma_regularizer=gamma_regularizer)

def Dense(units, activation, name=None):
    return keras.layers.Dense(units, name=name, activation=activation, kernel_initializer='glorot_normal', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())

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
    pool = GlobalAveragePooling2D()(x)
    pool = Reshape([1, 1, filters])(pool)
    squeeze = Dense(filters//ratio, activation='relu')(pool)
    excite = Dense(filters, activation='sigmoid')(squeeze)
    return multiply([x, excite])

def ValueHead(x, filters):
    out = ConvBlock(filters=32, kernel_size=1)(x)

    out = Flatten()(out)
    out = Dense(128, activation='relu')(out)

    out = Dense(1, name='value_head', activation='tanh')(out)
    return out

def PolicyHead(x, filters, output_size):
    out = ConvBlock(filters=32, kernel_size=1)(x)

    out = Flatten()(out)
    out = Dense(output_size, name='policy_head', activation='linear')(out)
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
    policy_head = PolicyHead(net, num_filters, output_size)

    if moves_left_size > 0:
        moves_left_head = MovesLeftHead(net, num_filters, moves_left_size)
        outputs = [value_head, policy_head, moves_left_head]
    else:
        outputs = [value_head, policy_head]

    model = Model(inputs=inputs,outputs=outputs)

    return model
