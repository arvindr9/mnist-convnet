#convnet in scrath from numpy(no tf for this file)
import numpy as np

with open("data/train-images-idx3-ubyte.gz") as img, open("data/train-labels-idx1-ubyte.gz") as lbl:
    
    X_train = np.fromfile(img, dtype=np.uint8)#.reshape(60000, 784)
    print(X_train.shape)
    print(X_train[: 30])
    Y_train = np.fromfile(lbl, dtype=np.uint8) 
    print(Y_train.shape)
