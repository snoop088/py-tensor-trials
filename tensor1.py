import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

tf.debugging.set_log_device_placement(True)

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0], dtype=float)
ys = np.array([-1.0, 5.0, 11.0, 17.0, 23.0, 29.0, 32.0], dtype=float)

model.fit(xs, ys, epochs=100)

print(model.predict([20.0]))