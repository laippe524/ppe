import numpy as np
import keras
import torch
from torch import Tensor
def SRMLayer(x:Tensor)->Tensor:
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = np.asarray(
                            [[filter1, filter1, filter1],
                             [filter2, filter2, filter2],
                             [filter3, filter3, filter3]])  # shape=(3,3,5,5)
    filters = np.transpose(filters, (2, 3, 1, 0))  # shape=(5,5,3,3)

    initializer_srm = keras.initializers.Constant(filters)
    output = keras.layers.Conv2D(3, (5, 5), padding='same', kernel_initializer=initializer_srm, use_bias=False, trainable=False)(x)
    return output

