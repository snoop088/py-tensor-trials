
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "../data/weights/inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)



pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss="binary_crossentropy", metrics=["acc"])

train_datagen = ImageDataGenerator(rescale=1./255.)
validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory('../data/horse-or-human/train', target_size=(150,150), class_mode='binary', batch_size=20)
validation_generator = validation_datagen.flow_from_directory('../data/horse-or-human/validation', target_size=(150,150), class_mode='binary')

history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=8,
            verbose=1)



test_path = '../data/horse-or-human/test/'
classify = ['pex-1.jpg', 'pex-2.jpg', 'pex-3.jpg', 'pex-4.jpg', 'pex-5.jpg', 'pex-6.jpg', 'pex-7.jpg']
for file in classify:
    img = image.load_img(test_path + file, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes[0])
    if classes[0] > 0.5:
        print(file + ' is a human')
    else:
        print(file + ' is a horse')


