import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import pdb
import preprocessing as pr
import matplotlib.pyplot as plt

def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy')
    plt.legend([ 'train','test','train_loss','test_loss'], loc='upper left')
    plt.savefig("model_performance.png")

def Model():
    tf.keras.backend.clear_session()
    inputs= tf.keras.Input(shape=(150,150,3))
    convo1= tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(inputs)
    maxpool1=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo1)
    convo2= tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(maxpool1)
    maxpool2=tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(convo2)
    #normalize=tf.keras.layers.BatchNormalization()(maxpool2)
    flatten=tf.keras.layers.Flatten()(maxpool2)
    dence1=tf.keras.layers.Dense(100,activation='relu')(flatten)
    dence2=tf.keras.layers.Dense(75,activation='relu')(dence1)
    dence3=tf.keras.layers.Dense(50,activation='relu')(dence2)
    dence4=tf.keras.layers.Dense(25,activation='relu')(dence3)
    dropout1=tf.keras.layers.Dropout(0.1)(dence4)
    dence5=tf.keras.layers.Dense(100,activation='relu')(flatten)
    dence6=tf.keras.layers.Dense(75,activation='relu')(dence5)
    dence7=tf.keras.layers.Dense(50,activation='relu')(dence6)
    dence8=tf.keras.layers.Dense(25,activation='relu')(dence7)
    dropout2=tf.keras.layers.Dropout(0.1)(dence8)
    #outputs=tf.keras.layers.Dense(10,activation='softmax')(dence1)
    conc=tf.keras.layers.Concatenate()([dropout1,dropout2])
    drop=tf.keras.layers.Dropout(0.5)(conc)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(drop)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

folders=os.listdir('Herb')
#pr.AUG()
#pr.Train_test_split()
datagen = ImageDataGenerator(rescale=1./255) 
training_set = datagen.flow_from_directory('train',target_size=(150,150),
                                                 batch_size = 10
                                                 ,classes=folders)

validation_set = datagen.flow_from_directory('test',target_size=(150,150),
                                            batch_size = 10,classes=folders)
tf.keras.backend.clear_session()
#model = tf.keras.models.Sequential([tf.keras.Input(shape=(150,150,3)),
#                                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
#                                    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
#                                    tf.keras.layers.Dropout(0.1),
#                                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
#                                    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
#                                    tf.keras.layers.Dropout(0.1),
#                                    tf.keras.layers.Flatten(),
#                                    tf.keras.layers.Dense(250,activation='relu'),
#                                    tf.keras.layers.Dense(10, activation='softmax')])
#
model=Model()
tf.keras.utils.plot_model(model, show_shapes=True,to_file='Model.jpg')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
model.compile(optimizer ='adam' , loss = 'categorical_crossentropy', metrics = ['acc'])
model.build(input_shape=(150,150))
history=model.fit(training_set, validation_data = validation_set, epochs = 100 ,steps_per_epoch=250 ,callbacks=[callback])
print('The final accuracy of the model against validtion set: ',history.history['val_acc'][-1]*100,'%')
plot(history)
model.save("Herbs_classifier.keras") 
