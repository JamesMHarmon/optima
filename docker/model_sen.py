
# https://towardsdatascience.com/understanding-residual-networks-9add4b664b03
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization, Input, merge, GlobalAveragePooling2D, multiply
from keras.layers.core import Activation, Layer
from keras.optimizers import Nadam

def Block(x, filters):
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)
    
    out = se_block(out, filters)

    out = keras.layers.add([x, out])
    out = LeakyReLU()(out)

    return out

def se_block(x, filters, ratio=16):
    out = GlobalAveragePooling2D()(x)
    out = Reshape([1, 1, filters])(out)
    out = Dense(filters//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(out)
    out = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(out)
    return multiply([x, out])

def ValueHead(x):
    # Value Head
    # https://github.com/glinscott/leela-chess/issues/47
    out = Conv2D(32, kernel_size=(1,1), activation='linear', use_bias=False)(x)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Flatten()(out)
    out = Dense(256, activation='linear')(out)
    out = LeakyReLU()(out)

    out = Dense(1, name='value_head', activation='tanh')(out)
    return out

def PolicyHead(x, filters, output_size):
    # Number of filters is half that of the residual layer's filter size.
    filters = int(filters / 2)
    out = Conv2D(filters, kernel_size=(1,1), activation='linear', use_bias=False)(x)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    out = Flatten()(out)
    out = Dense(output_size, name='policy_head', activation='softmax')(out)
    return out
    

def ResNet(num_filters, num_blocks, input_shape, output_size):
    inputs = Input(input_shape)
    net = Conv2D(filters=num_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False)(inputs)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    for _ in range(0, num_blocks):
        net = Block(net, num_filters)
    
    value_head = ValueHead(net)
    policy_head = PolicyHead(net, num_filters, output_size)

    model = Model(inputs=inputs,outputs=[value_head, policy_head])

    return model

def create_model(num_filters, num_blocks, input_shape, output_size):
    model = ResNet(num_filters, num_blocks, input_shape, output_size)
    return model
