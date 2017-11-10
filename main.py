#convnet in scrath from numpy(no tf for this file)
import numpy as np

with open("data/train-images-idx3-ubyte") as img, open("data/train-labels-idx1-ubyte") as lbl:
    
    X_train = np.fromfile(img, dtype=np.uint8)[16:].reshape(784, 60000)
    print(X_train.shape)
    print(X_train[0][500])
    Y_train = np.fromfile(lbl, dtype=np.uint8)[8:].reshape(1, 60000)
    print(Y_train.shape)
