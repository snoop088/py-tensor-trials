import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.965):
            print('Reached 96.5% accuracy so cancelling training')
            self.model.stop_training = True
        # return super().on_epoch_end(epoch, logs)
    
callbacks = myCallback()



data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=75, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

img_index = 37

classifications = model.predict(test_images)
for i, mm in enumerate(classifications[img_index]):
    print(i, ': ', mm)

# print(classifications[22])
print(test_labels[img_index])