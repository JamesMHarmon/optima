import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, ReLU, Input, GlobalAveragePooling2D, add, multiply, Concatenate, Cropping2D
from keras.layers.core import Activation, Layer
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers

DATA_FORMAT = 'channels_last'
ARIMAA_PIECE_MOVES = ['N','E','S','W','NN','NE','NW','EE','ES','SS','SW','WW','NNN','NNE','NNW','NEE','NWW','EEE','EES','ESS','SSS','SSW','SWW','WWW','NNNN','NNNE','NNNW','NNEE','NNWW','NEEE','NWWW','EEEE','EEES','EESS','ESSS','SSSS','SSSW','SSWW','SWWW','WWWW']
ARIMAA_PUSH_PULL_MOVES = ['NN','NE','NW','EN','EE','ES','SE','SS','SW','WN','WS','WW']

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
    
    out = SqueezeExcitation(out, filters, name=name)

    out = add([x, out])
    out = ReLU()(out)

    return out

def SqueezeExcitationWithBeta(x, filters, name, ratio=4):
    pool = GlobalAveragePooling2D(data_format=DATA_FORMAT, name=name + '/se/global_average_pooling2d')(x)
    squeeze = Dense(filters // ratio, activation='relu', name=name + '/se/1')(pool)
    excite_gamma = Reshape([1, 1, filters])(Dense(filters, activation='sigmoid', name=name + '/se/gamma')(squeeze))
    excite_beta = Reshape([1, 1, filters])(Dense(filters, activation=None, name=name + '/se/beta')(squeeze))

    return add([multiply([x, excite_gamma]), excite_beta])

def SqueezeExcitation(x, filters, name, ratio=4):
    pool = GlobalAveragePooling2D(data_format=DATA_FORMAT, name=name + '/se/global_average_pooling2d')(x)
    pool = Reshape([1, 1, filters])(pool)
    squeeze = Dense(filters // ratio, activation='relu', name=name + '/se/1')(pool)
    excite = Dense(filters, activation='sigmoid', name=name + '/se/2')(squeeze)
    return multiply([x, excite])

def ValueHead(x, filters):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='value_head')(x)

    out = Flatten()(out)
    out = Dense(filters, activation='relu', name='value_head/1')(out)
    out = Dense(1, activation='tanh', name='', full_name='value_head')(out)

    return out

def PolicyHeadFullyConnected(x, filters, output_size):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='policy_head')(x)

    out = Flatten()(out)
    out = Dense(output_size, activation=None, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name='', full_name='policy_head')(out)
    return out

def ArimaaPolicyHeadConvolutional(x, filters, output_size):
    conv_block = ConvBlock(filters=filters, kernel_size=3, name='policy_head')(x)

    def create_move_out(crop_dir, name):
        cropping = ((crop_dir.count('N'), crop_dir.count('S')), (crop_dir.count('W'), crop_dir.count('E')))
        move_conv = Conv2D(1, kernel_size=3, use_bias=True, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name=name)(conv_block)
        move_cropped = Cropping2D(cropping, data_format=DATA_FORMAT, name=name + '/cropping_2d')(move_conv)
        return Flatten()(move_cropped)

    def create_piece_move_out(dir):
        return create_move_out(crop_dir=dir, name='policy_head/move/' + dir)

    def create_push_pull_out(dir):
        crop_dir = dir[0] + dir[1].translate(str.maketrans('NESW', 'SWNE'))
        return create_move_out(crop_dir=crop_dir, name='policy_head/push_pull/' + dir)

    piece_move_outs = [create_piece_move_out(dir) for dir in ARIMAA_PIECE_MOVES]
    push_pull_outs = [create_push_pull_out(dir) for dir in ARIMAA_PUSH_PULL_MOVES]

    pass_out = ConvBlock(filters=2, kernel_size=1, name='policy_head/pass')(conv_block)
    pass_out = Flatten()(pass_out)
    pass_out = Dense(1, activation=None, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name='policy_head/pass/2')(pass_out)

    out = Concatenate(name='policy_head')(piece_move_outs + push_pull_outs + [pass_out])

    return out

def QuoridorPolicyHeadConvolutional(x, filters, output_size):
    conv_block = ConvBlock(filters=filters, kernel_size=3, name='policy_head')(x)

    def create_action_out(cropping, name):
        action_conv = Conv2D(1, kernel_size=3, use_bias=True, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name=name)(conv_block)
        action_cropped = action_conv if cropping is None else Cropping2D(cropping, data_format=DATA_FORMAT, name=name + '/cropping_2d')(action_conv)
        return Flatten()(action_cropped)

    pawn_move = create_action_out(None, name='policy_head/pawn_move')
    place_vertical = create_action_out(cropping=((1, 0), (0, 1)), name='policy_head/place_wall_vertical')
    place_horizontal = create_action_out(cropping=((1, 0), (0, 1)), name='policy_head/place_wall_horizontal')

    out = Concatenate(name='policy_head')([
        pawn_move,
        place_vertical,
        place_horizontal,
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

    if output_size == 2245:
        policy_head = ArimaaPolicyHeadConvolutional(net, num_filters, output_size)
    elif output_size == 209:
        policy_head = QuoridorPolicyHeadConvolutional(net, num_filters, output_size)
    else:
        policy_head = PolicyHeadFullyConnected(net, num_filters, output_size)

    if moves_left_size > 0:
        moves_left_head = MovesLeftHead(net, num_filters, moves_left_size)
        outputs = [value_head, policy_head, moves_left_head]
    else:
        outputs = [value_head, policy_head]

    model = Model(inputs=state_input,outputs=outputs)

    return model
