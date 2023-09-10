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
training_set = train_datagen.flow_from_directory('train',target_size=(400,400),
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
validation_set = test_datagen.flow_from_directory('test',target_size=(400,400),
                                            batch_size = 10,classes=os.listdir('data'))

tf.keras.backend.clear_session()
inputs=inputs = tf.keras.Input(shape=(400,400,3))
convo1= tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(inputs)
maxpool1=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo1)
convo2= tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(maxpool1)
maxpool2=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo2)
#normalize=tf.keras.layers.BatchNormalization()(maxpool2)
flatten=tf.keras.layers.Flatten()(maxpool2)
dropout1=tf.keras.layers.Dropout(0.5)(flatten)
dence1=tf.keras.layers.Dense(100,activation='relu')(dropout1)
dence2=tf.keras.layers.Dense(75,activation='relu')(dence1)
dence3=tf.keras.layers.Dense(50,activation='relu')(dence2)
dence4=tf.keras.layers.Dense(25,activation='relu')(dence3)
dropout2=tf.keras.layers.Dropout(0.5)(flatten)
dence5=tf.keras.layers.Dense(100,activation='relu')(dropout2)
dence6=tf.keras.layers.Dense(75,activation='relu')(dence5)
dence7=tf.keras.layers.Dense(50,activation='relu')(dence6)
dence8=tf.keras.layers.Dense(25,activation='relu')(dence7)
#outputs=tf.keras.layers.Dense(10,activation='softmax')(dence1)
conc=tf.keras.layers.Concatenate()([dence4,dence8])
outputs = tf.keras.layers.Dense(10, activation='softmax')(conc)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
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
