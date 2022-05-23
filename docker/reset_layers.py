import c4_model as c4
import numpy as np
from keras.utils.layer_utils import count_params

layers_to_reset = ['block_7', 'policy_head', 'value_head', 'moves_left_head']
layers_not_reset = []
layers_reset = []

c4.clear()
blank_model = c4.create(num_filters, num_blocks, (input_h, input_w, input_c), output_size, moves_left_size)
old_model = c4.load(source_model_path)

def save_weights_to_readable_file(model, file_name):
    with open(file_name, "w") as f:
        for i, layer in enumerate(model.layers):
            f.write(layer.name + " ")
            if count_params(layer.trainable_weights) > 0:
                weights = layer.get_weights()
                if len(weights) > 0:
                    app_weights = np.array([])
                    for weights in weights:
                        app_weights = np.append(app_weights, np.reshape(weights, -1))

                    np.savetxt(f, np.reshape(app_weights, (1, -1)), fmt='%1.4e')
                else:
                    f.write("\n")
            else:
                f.write("\n")

save_weights_to_readable_file(old_model, "weights_pre_reset.txt")

for i, layer in enumerate(old_model.layers):
    if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
        blank_layer = blank_model.layers[i]
        assert(blank_layer.name == layer.name)
        
        if layer.name.startswith(tuple(layers_to_reset)):
            blank_weights = blank_layer.get_weights()
            print('Resetting Weights for Layer', layer.name, np.shape(blank_weights))
            layer.set_weights(blank_weights)
            layers_reset.append(layer.name)
        else:
            layers_not_reset.append(layer.name)
            
save_weights_to_readable_file(old_model, "weights_post_reset.txt")
save_weights_to_readable_file(blank_model, "weights_blank.txt")

print("")            
print("RESET LAYERS:", layers_reset)
print("")            
print("SKIPPED LAYERS:", layers_not_reset)


old_model.save(target_model_path)

c4.export(target_model_path, export_model_path, num_filters, num_blocks, (input_h, input_w, input_c), output_size, moves_left_size)

