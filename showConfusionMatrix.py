import numpy as np
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

data_dir = pathlib.Path('chest_xray', 'thu nghiem 3')

batch_size = 32
img_height = 224
img_width = 224

model_path = pathlib.Path('model')

num_of_test_samples = 624
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    image_size=(img_height, img_width),
    shuffle=False,
    batch_size=32
)
class_names=test_ds.class_names
pneumonia_model = keras.models.load_model(os.path.join(model_path, 'vgg16_model.h5')) #Change to AlexNet, ResNes50

labels = ['NORMAL', 'PNEUMONIA']
y_test = []
for i in list(data_dir.glob(('test/NORMAL/*.jpeg'))):
    y_test.append(0)
for i in list(data_dir.glob(('test/PNEUMONIA/*.jpeg'))):
    y_test.append(1)
y_test = np.array(y_test)

predictions = pneumonia_model.predict(test_ds)
predictions = np.argmax(predictions, axis=1)
# optional: display 1D array of labels
# print(predictions)
# print(y_test)
cm = confusion_matrix(y_test,predictions)
print(cm)
print(classification_report(y_test, predictions, target_names = ['Normal (Class 0)', 'Pneumonia (Class 1)']))
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
plt.show()