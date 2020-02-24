import numpy as np
import keras
from keras.models import Model
from keras.layers import Reshape, Dense, Conv2D, Flatten, ReLU, BatchNormalization, Input, GlobalAveragePooling2D, multiply
from keras.layers.core import Activation, Layer
from keras.optimizers import Nadam
from keras import regularizers

def ConvBlock(x, filters, kernel_size=[3, 3]):
    out = Conv2D(filters, kernel_size=kernel_size, activation='linear', kernel_regularizer=l2_reg(), use_bias=False)(x)
    out = BatchNormalization(scale=False, beta_regularizer=l2_reg())(out)
    out = ReLU()(out)
    return out

def ResidualBlock(x, filters):
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_regularizer=l2_reg(), use_bias=False)(x)
    out = BatchNormalization(scale=False, beta_regularizer=l2_reg())(out)
    out = ReLU()(out)

    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_regularizer=l2_reg(), use_bias=False)(out)
    out = BatchNormalization(scale=True, beta_regularizer=l2_reg(), gamma_regularizer=l2_reg())(out)
    
    out = SqueezeExcitation(out, filters)

    out = keras.layers.add([x, out])
    out = ReLU()(out)

    return out

def SqueezeExcitation(x, filters, ratio=4):
    pool = GlobalAveragePooling2D()(x)
    pool = Reshape([1, 1, filters])(pool)
    squeeze = Dense(filters//ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(pool)
    excite = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(squeeze)
    return multiply([x, excite])

def ValueHead(x, filters):
    out = ConvBlock(x, filters=32, kernel_size=(1,1))

    out = Flatten()(out)
    out = Dense(128, activation='relu', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(out)

    out = Dense(1, name='value_head', activation='tanh', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(out)
    return out

def PolicyHead(x, filters, output_size):
    out = ConvBlock(x, filters=32, kernel_size=(1,1))

    out = Flatten()(out)
    out = Dense(output_size, name='policy_head', activation='linear', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(out)
    return out

def MovesLeftHead(x, filters, moves_left_size):
    out = ConvBlock(x, filters=8, kernel_size=(1,1))

    out = Flatten()(out)
    out = Dense(moves_left_size * 2, activation='relu', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(out)
    out = Dense(moves_left_size, name='moves_left_head', activation='softmax', kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())(out)
    return out

def ResNet(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    inputs = Input(input_shape)
    net = Conv2D(filters=num_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_regularizer=l2_reg(), use_bias=False)(inputs)
    net = BatchNormalization(scale=True, beta_regularizer=l2_reg(), gamma_regularizer=l2_reg())(net)
    net = ReLU()(net)

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

def create_model(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    model = ResNet(num_filters, num_blocks, input_shape, output_size, moves_left_size)
    return model

def l2_reg():
    return regularizers.l2(0.00005)
