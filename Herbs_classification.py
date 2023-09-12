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
import shutil

def Augment_data(img_path,Aug_dir):
    aug_img=[]
    image=Image.open(img_path)
    image=image.resize((400,400))
    width, height = image.size
    aug_img.append(np.array(ImageOps.grayscale(image)))
    aug_img.append(np.array(ImageOps.equalize(image)))
    image = np.array(image)
    aug_img.append(image)
    Noise = np.random.normal(0,10, (height, width,3)).astype(np.uint8)
    aug_img.append(image+Noise)
    aug_img.append((-1 * image + 255).astype(np.uint8))
    aug_img.append((5*image+255).astype(np.uint8))
    aug_img.append((cv2.GaussianBlur(image,(5,5),sigmaX=10,sigmaY=10)).astype(np.uint8))
    for i in range(len(aug_img)):
        plt.imsave(f'{Aug_dir}/{img_path.split("/")[-1]}{i}.jpg',aug_img[i])

folders=os.listdir('Herb')
os.mkdir('Aug_data')
for i in folders:
    os.mkdir(f'Aug_data/{i}')

for i in folders:
    files=os.listdir(f'Herb/{i}')
    for j in files:
        Augment_data(f'Herb/{i}/{j}',f'Aug_data/{i}')


folders=os.listdir('Aug_data')
os.mkdir('train')
os.mkdir('test')
for i in folders:
    os.mkdir(f'train/{i}')
    os.mkdir(f'test/{i}')
Train_files=[]
Test_files=[]
for i in folders:
    all_files=os.listdir(f'Aug_data/{i}')
    Train_files,Test_files=train_test_split(all_files,test_size=0.2)
    for j in Train_files:
        shutil.copy(f'Aug_data/{i}/{j}',f'train/{i}')
    for j in Test_files:
        shutil.copy(f'Aug_data/{i}/{j}',f'test/{i}')


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
