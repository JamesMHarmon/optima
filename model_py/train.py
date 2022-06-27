import os
import c4_model as c4
import logging as log
import tempfile
import tarfile
from json_file import JSONFile
from data_generator import DataGenerator
from replay_buffer import ReplayBuffer
from tensorboard_enriched import TensorBoardEnriched
from warmup_lr_scheduler import WarmupLearningRateScheduler
from fit_logger import FitLogger
from fit import Fit
from pyhocon import ConfigFactory

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

def export_bundle(model_dir, epoch, num_filters, num_blocks, input_h, input_w, input_c, policy_size, moves_left_size):
    export_model_dir = os.path.join(model_dir, 'exports')
    export_model_path = os.path.join(export_model_dir, f'{name(epoch + 1)}.tar.gz')
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

if __name__== '__main__':
    model_dir               = '/Arimaa_runs/run-2/model_8b96f'

    train_state_path        = os.path.join(model_dir, './train-state.json')
    model_info_path         = os.path.join(model_dir, './model-options.json')
    model_info_path         = os.path.join(model_dir, './model-info.json')
    config_path             = os.path.join(model_dir, './train.conf')
    tensor_board_path       = os.path.join(model_dir, './tensorboard')
    
    conf = ConfigFactory.parse_file(config_path)

    export_dir              = os.path.join(model_dir, conf.get_string('export_dir'))
    games_dir               = os.path.join(model_dir, conf.get_string('games_dir'))
    model_name              = conf.get_string('model_name')
    mode                    = conf.get_string('mode')
    
    min_visits              = conf.get_int('min_visits')
    games_per_epoch         = conf.get_int('games_per_epoch')
    window_size             = conf.get_int('window_size')
    avg_num_samples_per_pos = conf.get_float('avg_num_samples_per_pos')
    window_warmup           = conf.get_float('window_warmup')
    warmup_steps            = conf.get_int('warmup_steps')
    
    max_grad_norm           = conf.get_float('max_grad_norm')
    batch_size              = conf.get_float('batch_size')
    learning_rate           = conf.get_float('learning_rate')
    policy_loss_weight      = conf.get_float('policy_loss_weight')
    value_loss_weight       = conf.get_float('value_loss_weight')
    moves_left_loss_weight  = conf.get_float('moves_left_loss_weight')
    step_ratio              = conf.get_float('step_ratio')
    input_h                 = conf.get_int('input_h')
    input_w                 = conf.get_int('input_w')
    input_c                 = conf.get_int('input_c')
    policy_size             = conf.get_int('policy_size')
    moves_left_size         = conf.get_int('moves_left_size')
    num_filters             = conf.get_int('num_filters')
    num_blocks              = conf.get_int('num_blocks')
    min_visits              = conf.get_int('min_visits')

    def name(epoch):
        return '{0}_{1:05d}'.format(model_name, epoch)

    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    log.info(f'Games dir: {games_dir}')

    input_size = input_h * input_w * input_c
    value_size = 1
    train_ratio = 1.0

    initial_step, epoch = load_train_state(train_state_path)

    c4.clear()
    last_model_path = os.path.join(model_dir, 'models', name(epoch) + '.h5')
    log.info(f'Loading model: {last_model_path}')
    model = c4.load(last_model_path)

    lr_schedule = WarmupLearningRateScheduler(lr=learning_rate, warmup_steps=warmup_steps)
    accuracy_metrics = c4.metrics(model)
    
    buffer_cache_dir = os.path.realpath(os.path.join(model_dir, '..', 'replay_buffer_cache'))
    log.info(f'Replay Buffer cache: {buffer_cache_dir}')
    replay_buffer = ReplayBuffer(games_dir, min_visits, mode, cache_dir=buffer_cache_dir)

    while True:
        data_generator = DataGenerator(
            replay_buffer=replay_buffer,
            epoch=epoch,
            batch_size=batch_size,
            games_per_epoch=games_per_epoch,
            window_size=window_size,
            avg_num_samples_per_pos=avg_num_samples_per_pos,
            window_warmup=window_warmup,
            shapes={
                'X': (input_h, input_w, input_c),
                'yp': (policy_size,),
                'yv': (value_size,),
                'ym': (moves_left_size,)
            })

        tensor_board = TensorBoardEnriched(log_dir=tensor_board_path, step_ratio=step_ratio)
        fit_logger = FitLogger(log_steps=10)

        c4.compile(
            model,
            learning_rate=learning_rate,
            policy_loss_weight=policy_loss_weight,
            value_loss_weight=value_loss_weight,
            moves_left_loss_weight=moves_left_loss_weight,
        )
        
        log.info(f'Training {data_generator.__len__()} steps')

        steps = Fit(
            model=model,
            data_generator=data_generator,
            batch_size=batch_size,
            initial_epoch=epoch,
            initial_step=initial_step,
            train_size=train_ratio,
            clip_norm=max_grad_norm,
            accuracy_metrics=accuracy_metrics,
            callbacks=[lr_schedule, tensor_board, fit_logger]
        ).fit()
        
        initial_step = initial_step + steps

        model_path = os.path.join(model_dir, 'models', name(epoch + 1) + '.h5')
        log.info(f'Saving model: {model_path}')
        model.save(model_path)

        log.info('Saving Train State')
        save_train_state(train_state_path=train_state_path, steps=initial_step, epochs=epoch)

        export_bundle(model_dir, epoch, num_filters, num_blocks, input_h, input_w, input_c, policy_size, moves_left_size)

        epoch += 1

    