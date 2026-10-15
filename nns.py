import numpy as np
import tensorflow as tf
x = np.array([[200.0, 17,0]])
# normalization code 
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(x)
xn = norm_l(x)
layer_1 = tf.keras.layers.Dense(units=25, activation='sigmoid')
layer_2 = tf.keras.layers.Dense(units=15, activation='sigmoid')
layer_3 = tf.keras.layers.Dense(units=1, activation='sigmoid')
model = tf.keras.Sequential([layer_1, layer_2, layer_3])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, np.array([[1.0]]), epochs=1000, verbose=0)
prediction = model.predict(x)