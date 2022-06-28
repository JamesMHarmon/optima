import os
import c4_model as c4
import json
import logging as log
from train_utils import copy_bundle_to_export, export_bundle
from pyhocon import ConfigFactory

if __name__== '__main__':
    model_dir               = '/Arimaa_runs/run-2/model_8b96f' # os.getcwd()

    train_state_path        = os.path.join(model_dir, './train-state.json')
    model_info_path         = os.path.join(model_dir, './model-options.json')
    model_info_path         = os.path.join(model_dir, './model-info.json')
    config_path             = os.path.join(model_dir, './train.conf')
    tensor_board_path       = os.path.join(model_dir, './tensorboard')
    
    conf = ConfigFactory.parse_file(config_path)

    export_dir              = os.path.join(model_dir, conf.get_string('export_dir'))
    games_dir               = os.path.join(model_dir, conf.get_string('games_dir'))
    game                    = conf.get_string('game')
    run_name                = conf.get_string('run_name')
    model_name              = conf.get_string('model_name')
    mode                    = conf.get_string('mode')
    
    input_h                 = conf.get_int('input_h')
    input_w                 = conf.get_int('input_w')
    input_c                 = conf.get_int('input_c')
    policy_size             = conf.get_int('policy_size')
    moves_left_size         = conf.get_int('moves_left_size')
    num_filters             = conf.get_int('num_filters')
    num_blocks              = conf.get_int('num_blocks')

    def name(epoch):
        return '{0}_{1:05d}'.format(model_name, epoch)

    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    log.info(f'Games dir: {games_dir}')

    input_size = input_h * input_w * input_c
    input_shape = (input_h, input_w, input_c)
    value_size = 1
    train_ratio = 1.0
    
    model = c4.create(num_filters, num_blocks, input_shape, policy_size, moves_left_size)
    
    model_info_path = os.path.join(model_dir, 'model-info.json')
    with open(model_info_path, 'w') as f:
        json.dump({ 'game_name': game, 'run_name': run_name, 'model_num': 1 }, f, indent = 4)
        
    model_options_path = os.path.join(model_dir, 'model-options.json')
    with open(model_options_path, 'w') as f:
        json.dump({
            'num_filters': num_filters,
            'num_blocks': num_blocks,
            'channel_height': input_h,
            'channel_width': input_w,
            'channels': input_c,
            'output_size': policy_size,
            'moves_left_size': moves_left_size
        }, f, indent = 4)
    
    model_name_w_num = name(1)
    models_dir = os.path.join(model_dir, 'models')
    model_path = os.path.join(models_dir, model_name_w_num + '.h5.gz')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    log.info(f'Saving model: {model_path}')
    c4.save(model, model_path)

    export_bundle(model_dir, model_path, model_name_w_num, 0, num_filters, num_blocks, input_h, input_w, input_c, policy_size, moves_left_size)

    copy_bundle_to_export(conf, model_dir, export_dir, model_name_w_num)
