import numpy as np
import keras
from keras.models import Model
from keras.layers import Reshape, LeakyReLU as ReLU, Input, GlobalAveragePooling2D, add, multiply, Concatenate, Cropping2D
from keras.layers.core import Activation, Layer
from keras.optimizers import Nadam
from keras import regularizers

DATA_FORMAT = 'channels_last'

def l2_reg():
    return regularizers.l2(3e-5)

def l2_reg_policy():
    return regularizers.l2(1e-4)

def Flatten():
    return keras.layers.Flatten(data_format=DATA_FORMAT)

def Conv2D(filters, kernel_size, name, use_bias=False, bias_regularizer=None, kernel_regularizer=l2_reg()):
    return keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='glorot_normal',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_bias=use_bias,
        data_format=DATA_FORMAT,
        name=name + '/conv2d')

def BatchNorm(scale, name):
    return keras.layers.BatchNormalization(
        scale=scale,
        epsilon=1e-5,
        name=name + '/bn')

def Dense(units, name, activation, full_name=None, bias_regularizer=None, kernel_regularizer=l2_reg()):
    return keras.layers.Dense(
        units,
        activation=activation,
        kernel_initializer='glorot_normal',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name= full_name if full_name is not None else name + '/dense')

def ConvBlock(filters, kernel_size, name, batch_scale=False):
    def block(x):
        out = Conv2D(filters=filters, kernel_size=kernel_size, name=name + '/conv_block')(x)
        out = BatchNorm(scale=batch_scale, name=name + '/conv_block')(out)
        out = ReLU(name=name + '/conv_block/re_lu')(out)

        return out
    return block 

def ResidualBlock(x, filters, name):
    out = Conv2D(filters=filters, kernel_size=3, name=name + '/residual_block/1')(x)
    out = BatchNorm(scale=False, name=name + '/residual_block/1')(out)
    out = ReLU()(out)

    out = Conv2D(filters=filters, kernel_size=3, name=name + '/residual_block/2')(out)
    out = BatchNorm(scale=True, name=name + '/residual_block/2')(out)
    
    # out = SqueezeExcitation(out, filters, name=name)

    out = add([x, out])
    out = ReLU()(out)

    return out

def SqueezeExcitationWithBeta(x, filters, name, ratio=4):
    pool = GlobalAveragePooling2D(data_format=DATA_FORMAT, name=name + '/se/global_average_pooling2d')(x)
    squeeze = Dense(filters // ratio, activation=None, name=name + '/se/1')(pool)
    squeeze = ReLU(squeeze)
    excite_gamma = Reshape([1, 1, filters])(Dense(filters, activation='sigmoid', name=name + '/se/gamma')(squeeze))
    excite_beta = Reshape([1, 1, filters])(Dense(filters, activation=None, name=name + '/se/beta')(squeeze))

    return add([multiply([x, excite_gamma]), excite_beta])

def SqueezeExcitation(x, filters, name, ratio=4):
    pool = GlobalAveragePooling2D(data_format=DATA_FORMAT, name=name + '/se/global_average_pooling2d')(x)
    pool = Reshape([1, 1, filters])(pool)
    squeeze = Dense(filters // ratio, activation=None, name=name + '/se/1')(pool)
    squeeze = ReLU(squeeze)
    excite = Dense(filters, activation='sigmoid', name=name + '/se/2')(squeeze)
    return multiply([x, excite])

def ValueHead(x, filters):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='value_head')(x)

    out = Flatten()(out)
    out = Dense(filters, activation=None, name='value_head/1')(out)
    out = ReLU()(out)
    out = Dense(1, activation='tanh', name='', full_name='value_head')(out)

    return out

def PolicyHeadFullyConnected(x, filters, output_size):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='policy_head')(x)

    out = Flatten()(out)
    out = Dense(output_size, activation=None, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name='', full_name='policy_head')(out)
    return out

def PolicyHeadConvolutional(x, filters, output_size):
    conv_block = ConvBlock(filters=filters, kernel_size=3, name='policy_head')(x)

    def create_move_dir_out(cropping, name):
        move_dir_conv = Conv2D(1, kernel_size=3, use_bias=True, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name=name)(conv_block)
        move_dir_cropped = Cropping2D(cropping, data_format=DATA_FORMAT, name=name + '/cropping_2d')(move_dir_conv)
        return Flatten()(move_dir_cropped)

    move_up_out = create_move_dir_out(cropping=((1, 0), (0, 0)), name='policy_head/up')
    move_right_out = create_move_dir_out(cropping=((0, 0), (0, 1)), name='policy_head/right')
    move_down_out = create_move_dir_out(cropping=((0, 1), (0, 0)), name='policy_head/down')
    move_left_out = create_move_dir_out(cropping=((0, 0), (1, 0)), name='policy_head/left')

    pass_out = ConvBlock(filters=2, kernel_size=1, name='policy_head/pass')(conv_block)
    pass_out = Flatten()(pass_out)
    pass_out = Dense(1, activation=None, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name='policy_head/pass/2')(pass_out)

    out = Concatenate(name='policy_head')([
        move_up_out,
        move_right_out,
        move_down_out,
        move_left_out,
        pass_out
    ])

    return out

def MovesLeftHead(x, filters, moves_left_size):
    out = ConvBlock(filters=4, kernel_size=1, name='moves_left_head')(x)

    out = Flatten()(out)
    out = Dense(moves_left_size, name='', full_name='moves_left_head', activation='softmax')(out)
    return out

def create_model(num_filters, num_blocks, input_shape, output_size, moves_left_size):
    state_input = Input(shape=input_shape)
    net = ConvBlock(filters=num_filters, kernel_size=3, batch_scale=True, name='input')(state_input)

    for block_num in range(0, num_blocks):
        net = ResidualBlock(net, num_filters, name='block_' + str(block_num))

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

    model = Model(inputs=state_input,outputs=outputs)

    return model
