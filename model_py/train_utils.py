import glob
import shutil
import c4_model as c4
import logging as log
import os
import tempfile
import tarfile
from json_file import JSONFile
from model_sen import ModelDimensions

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

def export_bundle(model_dir, model_path, model_name: str, epoch: int, model_dims: ModelDimensions):
    export_model_dir = os.path.join(model_dir, 'exports')
    export_model_path = os.path.join(export_model_dir, f'{model_name}.tar.gz')
    log.info(f'Exporting model: {export_model_path}')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_export_path = os.path.join(tmpdirname, 'model')
        c4.export(model_path, tmp_export_path, model_dims)

        model_info_path = os.path.join(model_dir, 'model-info.json')
        model_options_path = os.path.join(model_dir, 'model-options.json')
        
        JSONFile(model_info_path).save_merge({ 'model_num': epoch + 1})

        if not os.path.isdir(export_model_dir):
            os.makedirs(export_model_dir)

        with tarfile.open(export_model_path, "w:gz") as tar:
            tar.add(tmp_export_path, arcname='model')
            tar.add(model_info_path, arcname='model-info.json')
            tar.add(model_options_path, arcname='model-options.json')

def copy_bundle_to_export(model_dir, export_dir, model_name):
    shutil.copy(
        os.path.join(model_dir, 'exports', f'{model_name}.tar.gz'),
        os.path.join(export_dir, f'{model_name}.tar.gz')
    )


