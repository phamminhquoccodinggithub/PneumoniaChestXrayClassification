import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import tensorflow as tf
from tensorflow import keras
model_path = pathlib.Path('model')

data_dir = pathlib.Path('chest_xray', 'thu nghiem 3')

labels = ['PNEUMONIA', 'NORMAL']
batch_size = 32
img_height = 224
img_width = 224

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    image_size=(img_height, img_width),
    shuffle=False,
    batch_size=32
)
class_names = test_ds.class_names
pneumonia_model = keras.models.load_model(os.path.join(model_path, 'resnet50.h5')) #Change to another model

_, accuracy = pneumonia_model.evaluate(test_ds, verbose=1)
print('Accuracy on the test set: %.2f'%(accuracy*100))

test_img_fn = os.path.join(data_dir, 'test/NORMAL/IM-0011-0001-0002.jpeg')
img = keras.preprocessing.image.load_img(
    test_img_fn, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prob = pneumonia_model.predict(img_array)
predict_class = class_names[np.argmax(prob)]
print('Predicted class: ', predict_class, 'Probability = ', np.max(prob), )
