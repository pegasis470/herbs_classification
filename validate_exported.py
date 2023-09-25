import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import tensorflow as tf
import pdb
folders=['Sirih', 'Nangka', 'Belimbing Wuluh', 'Kemangi', 'Jeruk Nipis', 'Jambu Biji', 'Seledri', 'Lidah Buaya', 'Pandan', 'Pepaya']
model=tf.keras.models.load_model("working.keras")

datagen = ImageDataGenerator(rescale=1./255)
validation=datagen.flow_from_directory('validation',target_size=(250,250),batch_size = 10,classes=folders)
pdb.set_trace()
_,accuracy = model.evaluate(validation)
print(f'Accuracy: {accuracy * 100:.2f}%')
predictions = model.predict(validation)
true_labels = validation.labels
predicted_labels = np.argmax(predictions, axis=0)
accuracy = np.mean(predicted_labels == true_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
