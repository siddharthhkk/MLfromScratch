import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import numpy as np
import tensorflow as tf
# normalization code 
x = np.array([[200.0, 17, 0]])
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(x)
xn = norm_l(x)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Dense(a_in, W, b):
    z = np.matmul(a_in, W) + b
    a_out = sigmoid(z)

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = Dense(x, 25, W1, b1)
    a2 = Dense(a1, 15, W2, b2)
    a3 = Dense(a2, 1, W3, b3)
    return a3
