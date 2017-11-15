import numpy as np
import matplotlib.pyplot as plt

def load_data():
    #convnet in scrath from numpy(no tf for this file) 

    with open("data/train-images-idx3-ubyte") as img, open("data/train-labels-idx1-ubyte") as lbl:
        
        X_train = np.fromfile(img, dtype=np.uint8)[16:].reshape(784, 60000)
        print(X_train.shape)
        print(X_train[0][500])
        Y_train = np.fromfile(lbl, dtype=np.uint8)[8:].reshape(1, 60000)
        print(Y_train.shape)
        return X_train, Y_train

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'çonstant', constant_values = 0)
    return X_pad
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev-f+2*pad)/stride) + 1
    n_W = int((n_W_prev-f+2*pad)/stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    cache = (A_prev, W, b, parameters)

    return Z, cache

X_train, Y_train = load_data()




