import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
folders=os.listdir('Herb')
model=keras.models.load_model("Herbs_classifier.keras")
datagen = ImageDataGenerator(rescale=1./255)
validation=datagen.flow_from_directory('validation',target_size=(150,150),
                                            batch_size = 20,classes=folders)
predictions=model.predict(validation)
true_labels = validation.classes
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

