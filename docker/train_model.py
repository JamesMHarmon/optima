import os
import json
from keras.callbacks import TensorBoard
import c4_model as c4

if __name__== "__main__":

    source_model_path   = os.environ['SOURCE_MODEL_PATH']
    target_model_path   = os.environ['TARGET_MODEL_PATH']
    export_model_path   = os.environ['EXPORT_MODEL_PATH']
    tensor_board_path   = os.environ['TENSOR_BOARD_PATH']

    data_paths          = os.environ['DATA_PATHS'].split(',')
    train_ratio         = float(os.environ['TRAIN_RATIO'])
    train_batch_size    = int(os.environ['TRAIN_BATCH_SIZE'])
    epochs              = int(os.environ['EPOCHS'])
    initial_epoch       = int(os.environ['INITIAL_EPOCH'])
    learning_rate       = float(os.environ['LEARNING_RATE'])
    policy_loss_weight  = float(os.environ['POLICY_LOSS_WEIGHT'])
    value_loss_weight   = float(os.environ['VALUE_LOSS_WEIGHT'])

    input_h             = int(os.environ['INPUT_H'])
    input_w             = int(os.environ['INPUT_W'])
    input_c             = int(os.environ['INPUT_C'])
    output_size         = int(os.environ['OUTPUT_SIZE'])
    num_filters         = int(os.environ['NUM_FILTERS'])
    num_blocks          = int(os.environ['NUM_BLOCKS'])

    print(data_paths)
    c4.clear()
    
    i = 0
    X = []
    yv = []
    yp = []
    first_run = True
    while i < len(data_paths):
        path = data_paths[i]
        with open(path, "r") as read_file:
            print("Loading Data: " + path)
            data = json.load(read_file)
            X = X + data["x"]
            yv = yv + data["yv"]
            yp = yp + data["yp"]

        if ((i + 1) % 10 == 0 or i == len(data_paths) - 1):
            model = c4.load_model(source_model_path)

            tensor_board = TensorBoard(log_dir=tensor_board_path,update_freq='epoch')
            callbacks = [tensor_board] if first_run else []

            c4.train(model, X, yv, yp, train_ratio, train_batch_size, epochs, initial_epoch, learning_rate, policy_loss_weight, value_loss_weight, callbacks)

            X = []
            yv = []
            yp = []
            first_run = False

        i += 1

    model.save(target_model_path)

    c4.export(target_model_path, export_model_path, num_filters, num_blocks, (input_h, input_w, input_c), output_size)
