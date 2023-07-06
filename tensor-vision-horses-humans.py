import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory('../data/horse-or-human/train', target_size=(300,300), class_mode='binary')
validation_generator = validation_datagen.flow_from_directory('../data/horse-or-human/validation', target_size=(300,300), class_mode='binary')

# data = tf.keras.datasets.fashion_mnist

# (training_images, training_labels), (test_images, test_labels) = data.load_data()

# training_images = training_images.reshape(60000, 28, 28, 1)
# training_images = training_images / 255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(768, activation='relu'), 
                                    tf.keras.layers.Dense(1, activation='sigmoid')])
print(model.summary())
model.compile(optimizer=RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

test_path = '../data/horse-or-human/test/'
classify = ['pex-1.jpg', 'pex-2.jpg', 'pex-3.jpg', 'pex-4.jpg', 'pex-5.jpg', 'pex-6.jpg', 'pex-7.jpg']
for file in classify:
    img = image.load_img(test_path + file, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes[0])
    if classes[0] > 0.5:
        print(file + ' is a human')
    else:
        print(file + ' is a horse')





# img_index = 99

# classifications = model.predict(test_images)
# print(model.count_params())
# for i, mm in enumerate(classifications[img_index]):
#     print(i, ': ', mm)

# # print(classifications[22])
# print(test_labels[img_index])
