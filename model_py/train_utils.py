import glob
import shutil
import c4_model as c4
import logging as log
import os
import tempfile
import tarfile
from json_file import JSONFile

def load_train_state(train_state_path):
    train_state = JSONFile(train_state_path).load_or_save_defaults({ 'steps': 0, 'epochs': 0 })

    initial_step = int(train_state['steps'] + 1)
    epoch = int(train_state['epochs'] + 1)
        
    return initial_step, epoch

def save_train_state(train_state_path, steps, epochs):
    train_state = {}

    train_state['steps'] = steps
    train_state['epochs'] = epochs

    JSONFile(train_state_path).save_merge(train_state)

def export_bundle(model_dir, model_path, model_name, epoch, num_filters, num_blocks, input_h, input_w, input_c, policy_size, moves_left_size):
    export_model_dir = os.path.join(model_dir, 'exports')
    export_model_path = os.path.join(export_model_dir, f'{model_name}.tar.gz')
    log.info(f'Exporting model: {export_model_path}')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_export_path = os.path.join(tmpdirname, 'model')
        c4.export(model_path, tmp_export_path, num_filters, num_blocks, (input_h, input_w, input_c), policy_size, moves_left_size)

        model_info_path = os.path.join(model_dir, 'model-info.json')
        model_options_path = os.path.join(model_dir, 'model-options.json')
        
        JSONFile(model_info_path).save_merge({ 'model_num': epoch + 1})

        if not os.path.isdir(export_model_dir):
            os.makedirs(export_model_dir)

        with tarfile.open(export_model_path, "w:gz") as tar:
            tar.add(tmp_export_path, arcname='model')
            tar.add(model_info_path, arcname='model-info.json')
            tar.add(model_options_path, arcname='model-options.json')

def copy_bundle_to_export(conf, model_dir, export_dir, model_name):
    place_dir = conf.get('arimaa_place_dir', None)

    # Special override for the game of arimaa which bundles two models together.
    if place_dir is None:
        shutil.copy(
            os.path.join(model_dir, 'exports', f'{model_name}.tar.gz'),
            os.path.join(export_dir, f'{model_name}.tar.gz')
        )
    else:
        match_arimaa_place_model(conf, place_dir, model_dir, export_dir, model_name)

def match_arimaa_place_model(conf, place_dir, model_dir, export_dir, model_name):
    place_dir = conf.get('arimaa_place_dir')

    place_exports_dir = str(os.path.abspath(os.path.join(model_dir, place_dir, 'exports')))
    list_of_place_models = glob.glob(place_exports_dir + '/*.tar.gz')
    latest_place_model = max(list_of_place_models, key=os.path.getctime)
    
    log.info('Combining with place model: ', latest_place_model)

    export_model_path = os.path.join(export_dir, f'{model_name}.tar.gz')
    play_model_path = os.path.join(model_dir, 'exports', f'{model_name}.tar.gz')
    with tarfile.open(export_model_path, "w:gz") as tar:
        tar.add(play_model_path, arcname='play.tar.gz')
        tar.add(latest_place_model, arcname='place.tar.gz')

