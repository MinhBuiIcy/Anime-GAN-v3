import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model


def build_vgg19(base_model, layer_names):
    # Get the outputs of the specified layers
    layer_outputs = [
        base_model.get_layer(layer_name).output if isinstance(layer_name, str)
        else base_model.layers[layer_name].output
        for layer_name in layer_names
    ]
    # Create a model with the same input as the base model and RaggedTensor outputs
    multi_layer_model = Model(inputs=base_model.input, outputs=layer_outputs)

    return multi_layer_model
