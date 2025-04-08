from dataclasses import dataclass
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, ReLU, Input, GlobalAveragePooling2D, add, multiply
from tensorflow.keras import regularizers
from tensorflow.keras import layers as keras_layers

DATA_FORMAT = 'channels_last'

@dataclass()
class InputDimensions:
    input_h: int
    input_w: int
    input_c: int

@dataclass()
class ModelDimensions:
    num_filters: int
    num_blocks: int
    policy_size: int
    moves_left_size: int
    input_dims: InputDimensions

def l2_reg():
    return regularizers.l2(1e-5)

def l2_reg_policy():
    return regularizers.l2(4e-5)

def Flatten():
    return keras_layers.Flatten(data_format=DATA_FORMAT)

def Conv2D(filters, kernel_size, name, use_bias=False, bias_regularizer=None, kernel_regularizer=l2_reg()):
    return keras_layers.Conv2D(
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
    return keras_layers.BatchNormalization(
        scale=scale,
        epsilon=1e-5,
        name=name + '/bn')

def Dense(units, name, activation, full_name=None, bias_regularizer=None, kernel_regularizer=l2_reg()):
    return keras_layers.Dense(
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

def VictoryMarginHead(x, filters):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='victory_margin_head')(x)

    out = Flatten()(out)
    out = Dense(filters, activation='relu', name='victory_margin_head/1')(out)
    out = Dense(1, activation=None, name='', full_name='victory_margin_head')(out)

    return out

def PolicyHeadFullyConnected(x, filters, output_size):
    out = ConvBlock(filters=filters // 8, kernel_size=1, name='policy_head')(x)

    out = Flatten()(out)
    out = Dense(output_size, activation=None, bias_regularizer=l2_reg_policy(), kernel_regularizer=l2_reg_policy(), name='', full_name='policy_head')(out)
    return out

def MovesLeftHead(x, filters, moves_left_size):
    out = ConvBlock(filters=4, kernel_size=1, name='moves_left_head')(x)

    out = Flatten()(out)
    out = Dense(moves_left_size, name='', full_name='moves_left_head', activation='softmax')(out)
    return out

def create_model(model_dims: ModelDimensions, policy_head=None, victory_margin_head=False):
    num_filters, num_blocks, input_dims, output_size, moves_left_size = (
        model_dims.num_filters,
        model_dims.num_blocks,
        model_dims.input_dims,
        model_dims.policy_size,
        model_dims.moves_left_size
    )

    input_shape = (input_dims.input_h, input_dims.input_w, input_dims.input_c)
    state_input = Input(shape=input_shape)
    net = ConvBlock(filters=num_filters, kernel_size=3, batch_scale=True, name='input')(state_input)

    for block_num in range(0, num_blocks):
        net = ResidualBlock(net, num_filters, name='block_' + str(block_num))

    value_head = ValueHead(net, num_filters)

    if policy_head is None:
        policy_head = lambda net, num_filters: PolicyHeadFullyConnected(net, num_filters, output_size)

    policy_head = policy_head(net, num_filters)

    outputs = [value_head, policy_head]

    if moves_left_size > 0:
        moves_left_head = MovesLeftHead(net, num_filters, moves_left_size)
        outputs.append(moves_left_head)

    if victory_margin_head:
        victory_margin_head = VictoryMarginHead(net, num_filters)
        outputs.append(victory_margin_head)

    model = Model(inputs=state_input,outputs=outputs)

    return model
