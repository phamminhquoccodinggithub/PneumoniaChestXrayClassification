import os
import pathlib
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import PIL.Image
import PIL.ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

model_path = pathlib.Path('model')
data_dir = pathlib.Path('chest_xray', 'thu nghiem 3')
main = tk.Tk()
main.title("Pneumonia Dectection")
main.geometry("1280x720")
main["bg"] = "#856ff8"
main.attributes("-topmost", True)  # always in top
value = StringVar()
accuracy = DoubleVar()

def openFile(filename):
    img = PIL.Image.open(filename)
    img = img.resize((680, 560))
    img = PIL.ImageTk.PhotoImage(img)
    global photo
    photo = Label(main, image=img)
    photo.pack()
    photo.image = img
    photo = photo.place(x=25, y=70)

def clicked():
    global filename
    filename = filedialog.askopenfilename()
    openFile(filename)
    return filename


def predict():
    if (networkchoosen.get()=="VGG16"):
        pneumonia_model = keras.models.load_model(os.path.join(model_path, 'vgg16_model.h5'))
        img_size = 224
    elif (networkchoosen.get()=="AlexNet"):
        pneumonia_model = keras.models.load_model(os.path.join(model_path, 'alexnet_model.h5'))
        img_size = 227
    else:
        pneumonia_model = keras.models.load_model(os.path.join(model_path, 'resnet50.h5'))
        img_size = 224
    test_img_fn = os.path.join(filename)
    img = keras.preprocessing.image.load_img(
        test_img_fn, target_size=(img_size, img_size)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prob = pneumonia_model.predict(img_array)
    pred = np.argmax(prob, axis=1)
    if (pred == 0):
        pred_class = 'NORMAL'
    else:
        pred_class = 'PNEUMONIA'
    accuracy=str(round(np.max(prob)*100, 2))
    value.set(value=pred_class+' with '+accuracy+'%')


def reset():
    value.set('')

but1 = Button(main, text="Insert Picture", font=("Times New Roman", 13), command=clicked)
but1.pack()
but1.place(x=300, y=650)
but2 = Button(main, text="Predict", font=("Times New Roman", 13), command=predict)
but2.place(x=920, y=130)
but3 = Button(main, text="Reset", font=("Times New Roman", 13), command=reset)
but3.place(x=1000, y=130)

T1 = Text(main, bg="white", height=35, width=85)
T1.pack()
T1.place(x=25, y=70)

lb1 = Label(main, text="Picture", bg="Black", font=("Times New Roman", 13), fg="White")
lb1.place(x=325, y=30)
lb2 = Label(main, text="Result", bg="Black", font=("Times New Roman", 13), fg="white")
lb2.place(x=800, y=200)

T2 = Text(main, bg="White", height=2.8, width=50)
T2.place(x=800, y=230)
combobox = Label(main, text="Select the neural network",
          font=("Times New Roman", 13))
combobox.pack( pady=30)
combobox.place(x=770, y=30)

n = tk.StringVar()
networkchoosen = ttk.Combobox(main, width=15,
                              textvariable=n)
networkchoosen['values'] = ('VGG16',
                            'AlexNet',
                            'ResNet50')

networkchoosen.current(0)
networkchoosen.place(x=800, y=60)

result = Label(main, textvariable=value, bg="white", font=(None, 30))
result = result.place(x=800, y=230)
main.mainloop()
