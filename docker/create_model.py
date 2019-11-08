import os
import tensorflow as tf

import c4_model as c4

if __name__== "__main__":

    target_model_path   = os.environ['TARGET_MODEL_PATH']
    export_model_path   = os.environ['EXPORT_MODEL_PATH']

    input_h             = int(os.environ['INPUT_H'])
    input_w             = int(os.environ['INPUT_W'])
    input_c             = int(os.environ['INPUT_C'])
    output_size         = int(os.environ['OUTPUT_SIZE'])
    num_filters         = int(os.environ['NUM_FILTERS'])
    num_blocks          = int(os.environ['NUM_BLOCKS'])

    c4.clear()
    model = c4.create(num_filters, num_blocks, (input_h, input_w, input_c), output_size)

    model.save(target_model_path)

    c4.export(target_model_path, export_model_path, num_filters, num_blocks, (input_h, input_w, input_c), output_size)
