import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split
import numpy as np
import keras.utils as image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest') 
training_set = train_datagen.flow_from_directory('train',target_size=(150,150),
                                                 batch_size = 10
                                                 ,classes=os.listdir('data'))
test_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
                                  
        horizontal_flip=True,
        fill_mode='nearest')
validation_set = test_datagen.flow_from_directory('test',target_size=(150,150),
                                            batch_size = 10,classes=os.listdir('data'))

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([tf.keras.Input(shape=(150,150,3)),
                                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100,activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
model.build(input_shape=(150,150))
history=model.fit(x = training_set, validation_data = validation_set, epochs = 15,steps_per_epoch=100)
print('The final accuracy of the model against validtion set: ',history.history['val_accuracy'][-1]*100,'%')
def plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy')
    plt.legend([ 'train','test','train_loss','test_loss'], loc='upper left')
    plt.savefig("loss.png")
    plt.show()
plot(history)
