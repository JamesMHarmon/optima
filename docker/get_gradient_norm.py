from tensorflow.keras import backend as K

def get_gradient_norm_func(model):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    summed_squares = [K.sum(K.square(g)) for g in grads]
    norm = K.sqrt(sum(summed_squares))
    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    func = K.function(inputs, norm)
    return func

def get_gradient_norm(model, X, y):
    gradient_norm_func = get_gradient_norm_func(model)
    data = model._standardize_user_data(X, y)
    grad_norm = gradient_norm_func(data)
    return grad_norm