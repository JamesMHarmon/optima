from tensorflow.keras.layers import Concatenate, Cropping2D

from model_sen import ConvBlock, Conv2D, DATA_FORMAT, l2_reg_policy, Flatten, Dense

ARIMAA_PIECE_MOVES = ['N','E','S','W','NN','NE','NW','EE','ES','SS','SW','WW','NNN','NNE','NNW','NEE','NWW','EEE','EES','ESS','SSS','SSW','SWW','WWW','NNNN','NNNE','NNNW','NNEE','NNWW','NEEE','NWWW','EEEE','EEES','EESS','ESSS','SSSS','SSSW','SSWW','SWWW','WWWW']
ARIMAA_PUSH_PULL_MOVES = ['NN','NE','NW','EN','EE','ES','SE','SS','SW','WN','WS','WW']

def get_policy_head_fn_by_policy_size(policy_size):
    if policy_size == 2245:
        policy_head_fn = lambda net, num_filters: ArimaaPolicyHeadConvolutional(net, num_filters)
    elif policy_size == 209:
        policy_head_fn = lambda net, num_filters: QuoridorPolicyHeadConvolutional(net, num_filters)
    else:
        policy_head_fn = None

    return policy_head_fn

def ArimaaPolicyHeadConvolutional(x, filters):
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

def QuoridorPolicyHeadConvolutional(x, filters):
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
