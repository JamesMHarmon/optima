import os
import json
import c4_model as c4
import numpy as np
from tensorboard_enriched import TensorBoardEnriched

if __name__== "__main__":

    source_model_path       = os.environ['SOURCE_MODEL_PATH']
    target_model_path       = os.environ['TARGET_MODEL_PATH']
    export_model_path       = os.environ['EXPORT_MODEL_PATH']
    tensor_board_path       = os.environ['TENSOR_BOARD_PATH']

    data_paths              = os.environ['DATA_PATHS'].split(',')
    train_ratio             = float(os.environ['TRAIN_RATIO'])
    train_batch_size        = int(os.environ['TRAIN_BATCH_SIZE'])
    epochs                  = int(os.environ['EPOCHS'])
    initial_epoch           = int(os.environ['INITIAL_EPOCH'])
    max_grad_norm           = float(os.environ['MAX_GRAD_NORM'])
    learning_rate           = float(os.environ['LEARNING_RATE'])
    policy_loss_weight      = float(os.environ['POLICY_LOSS_WEIGHT'])
    value_loss_weight       = float(os.environ['VALUE_LOSS_WEIGHT'])
    moves_left_loss_weight  = float(os.environ['MOVES_LEFT_LOSS_WEIGHT'])

    input_h                 = int(os.environ['INPUT_H'])
    input_w                 = int(os.environ['INPUT_W'])
    input_c                 = int(os.environ['INPUT_C'])
    output_size             = int(os.environ['OUTPUT_SIZE'])
    moves_left_size         = int(os.environ['MOVES_LEFT_SIZE'])
    num_filters             = int(os.environ['NUM_FILTERS'])
    num_blocks              = int(os.environ['NUM_BLOCKS'])

    input_size = input_h * input_w * input_c
    yv_size = 1

    print(data_paths)
    c4.clear()
    model = c4.load(source_model_path)
    tensor_board = TensorBoardEnriched(log_dir=tensor_board_path)

    for (i, path) in enumerate(data_paths):
        print("Loading Data: " + path)
        dataset = np.load(path).reshape(-1, input_size + output_size + moves_left_size + yv_size)
        X = dataset[:,0:input_size].reshape(dataset.shape[0],input_h,input_w,input_c)
        start_index = input_size
        yp = dataset[:,start_index:start_index + output_size]
        start_index += output_size
        yv = dataset[:,start_index]
        start_index += yv_size
        ym = dataset[:,start_index:]

        callbacks = [tensor_board] if i == 0 else []

        c4.train(model, X, yv, yp, ym, train_ratio, train_batch_size, epochs, initial_epoch + i, max_grad_norm, learning_rate, policy_loss_weight, value_loss_weight, moves_left_loss_weight, callbacks)

    model.save(target_model_path)

    c4.export(target_model_path, export_model_path, num_filters, num_blocks, (input_h, input_w, input_c), output_size, moves_left_size)
