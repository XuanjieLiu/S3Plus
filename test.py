import os
import sys

def make_multi_layers(layers_unit, num):
    layers = []
    for i in range(num):
        layers.extend(layers_unit)
    return layers

aa = make_multi_layers([1,2], 3)
print(bool(0.0001))