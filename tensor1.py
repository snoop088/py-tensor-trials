import tensorrt
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

print(tf.__version__)
tf.debugging.set_log_device_placement(True)
device_name = tf.test.gpu_device_name()
print(device_name)
model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0], dtype=float)
ys = np.array([-1.0, 5.0, 11.0, 17.0, 23.0, 29.0, 32.0], dtype=float)

model.fit(xs, ys, epochs=33)

print(model.predict([20.0]))
