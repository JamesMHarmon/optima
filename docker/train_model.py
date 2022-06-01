import os
import c4_model as c4
import json
from tensorboard_enriched import TensorBoardEnriched
from split_file_data_generator import SplitFileDataGenerator
from warmup_lr_scheduler import WarmupLearningRateScheduler
from fit import Fit

if __name__== "__main__":
    source_model_path       = os.environ['SOURCE_MODEL_PATH']
    target_model_path       = os.environ['TARGET_MODEL_PATH']
    export_model_path       = os.environ['EXPORT_MODEL_PATH']
    tensor_board_path       = os.environ['TENSOR_BOARD_PATH']
    train_state_path        = os.environ['TRAIN_STATE_PATH']

    data_paths              = os.environ['DATA_PATHS'].split(',')
    total_samples           = int(os.environ['TOTAL_SAMPLES'])
    train_ratio             = float(os.environ['TRAIN_RATIO'])
    train_batch_size        = int(os.environ['TRAIN_BATCH_SIZE'])
    epoch                   = int(os.environ['EPOCH'])
    max_grad_norm           = float(os.environ['MAX_GRAD_NORM'])
    learning_rate           = float(os.environ['LEARNING_RATE'])
    policy_loss_weight      = float(os.environ['POLICY_LOSS_WEIGHT'])
    value_loss_weight       = float(os.environ['VALUE_LOSS_WEIGHT'])
    moves_left_loss_weight  = float(os.environ['MOVES_LEFT_LOSS_WEIGHT'])
    step_ratio              = float(os.environ['STEP_RATIO'])

    input_h                 = int(os.environ['INPUT_H'])
    input_w                 = int(os.environ['INPUT_W'])
    input_c                 = int(os.environ['INPUT_C'])
    output_size             = int(os.environ['OUTPUT_SIZE'])
    moves_left_size         = int(os.environ['MOVES_LEFT_SIZE'])
    num_filters             = int(os.environ['NUM_FILTERS'])
    num_blocks              = int(os.environ['NUM_BLOCKS'])

    input_size = input_h * input_w * input_c
    value_size = 1
    train_ratio = 1.0

    c4.clear()
    model = c4.load(source_model_path)

    if epoch == 1 and not os.path.isfile(train_state_path):
        with open(train_state_path, 'w') as f:
            json.dump({ 'steps': 0 }, f, indent = 4)

    with open(train_state_path, 'r') as f:
        initial_step = int(json.load(f)['steps'] + 1)

    lr_schedule = WarmupLearningRateScheduler(lr=learning_rate, warmup_steps=1000)
    tensor_board = TensorBoardEnriched(log_dir=tensor_board_path, step_ratio=step_ratio)

    callbacks = [lr_schedule, tensor_board]

    generator = SplitFileDataGenerator(
        batch_size=train_batch_size,
        files=data_paths,
        total_samples=total_samples,
        input_size=input_size,
        output_size=output_size,
        moves_left_size=moves_left_size,
        value_size=value_size,
        input_h=input_h,
        input_w=input_w,
        input_c=input_c,
    )

    c4.compile(
        model=model,
        learning_rate=learning_rate,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        moves_left_loss_weight=moves_left_loss_weight
    )

    accuracy_metrics = c4.metrics(model)

    Fit(
        model=model,
        data_generator=generator,
        batch_size=train_batch_size,
        initial_epoch=epoch,
        initial_step=initial_step,
        train_size=train_ratio,
        clip_norm=max_grad_norm,
        accuracy_metrics=accuracy_metrics,
        callbacks=callbacks
    ).fit()

    model.save(target_model_path)

    with open(train_state_path, 'r') as f:
        train_state = json.load(f)

    with open(train_state_path, 'w') as f:
        train_state['steps'] = initial_step - 1 + int(total_samples / train_batch_size)
        json.dump(train_state, f, indent = 4)

    c4.export(target_model_path, export_model_path, num_filters, num_blocks, (input_h, input_w, input_c), output_size, moves_left_size)
