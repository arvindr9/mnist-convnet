from utils import *

def load_data():
    #convnet in scrath from numpy(no tf for this file) 

    with open("data/train-images-idx3-ubyte") as img, open("data/train-labels-idx1-ubyte") as lbl:
        
        X_train = np.fromfile(img, dtype=np.uint8)[16:].reshape(784, 60000)
        print(X_train.shape)
        print(X_train[0][500])
        Y_train = np.fromfile(lbl, dtype=np.uint8)[8:].reshape(1, 60000)
        print(Y_train.shape)
        return X_train, Y_train

np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2,
                "stride": 1}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
dA, dW, db = conv_backward(Z, cache_conv)
print(np.mean(dA), np.mean(dW), np.mean(db))