import torch
import torch.nn as nn


HIDDEN_PARAM = [1024, 1024, 1024]
IS_BATCH_NORM = False
IS_LAYER_NORM = True
NG_SLOPE = 0.01


def make_multi_layers(input_dim, output_dim, hidden_param, is_batch_norm=False, is_layer_norm=False):
    """
    Creates a neural network with multiple layers, allowing optional BatchNorm or LayerNorm.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output layer.
        hidden_param (list): List of hidden layer sizes.
        is_batch_norm (bool): If True, apply BatchNorm after each layer.
        is_layer_norm (bool): If True, apply LayerNorm after each layer.
        
    Returns:
        nn.Sequential: The constructed neural network as a sequential model.
    """
    def add_layer(layers, in_dim, out_dim):
        layers.append(nn.Linear(in_dim, out_dim))
        if is_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if is_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.LeakyReLU(negative_slope=NG_SLOPE))

    layers = []
    # Add first layer
    add_layer(layers, input_dim, hidden_param[0])
    # Add hidden layers
    for i in range(1, len(hidden_param)):
        add_layer(layers, hidden_param[i-1], hidden_param[i])
    # Add output layer (no activation or normalization)
    layers.append(nn.Linear(hidden_param[-1], output_dim))
    
    return nn.Sequential(*layers)


class FCProjector(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_param=HIDDEN_PARAM, 
                 is_batch_norm=IS_BATCH_NORM, 
                 is_layer_norm=IS_LAYER_NORM
                 ):
        super(FCProjector, self).__init__()
        self.linear = make_multi_layers(
            input_dim, output_dim, hidden_param, 
            is_batch_norm=is_batch_norm, is_layer_norm=is_layer_norm)


    def forward(self, x):
        return self.linear(x)