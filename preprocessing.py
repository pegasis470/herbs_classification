import keras.utils as image
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

folders=os.listdir('Herb')


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



def AUG():
    os.mkdir('Aug_data')
    for i in folders:
        os.mkdir(f'Aug_data/{i}')
    for i in folders:
        files=os.listdir(f'Herb/{i}')
        for j in files:
            Augment_data(f'Herb/{i}/{j}',f'Aug_data/{i}')
    

def Train_test_split():
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('validation')
    for i in folders:
        os.mkdir(f'train/{i}')
        os.mkdir(f'test/{i}')
        os.mkdir(f'validation/{i}')
    for i in folders:
        all_files=os.listdir(f'Aug_data/{i}')
        Train_files,Test_files=train_test_split(all_files,test_size=0.2)
        Train_files,val_files=train_test_split(Train_files,test_size=0.1)
        for j in val_files:
            shutil.copy(f'Aug_data/{i}/{j}',f'validation/{i}')
        for j in Train_files:
            shutil.copy(f'Aug_data/{i}/{j}',f'train/{i}')
        for j in Test_files:
            shutil.copy(f'Aug_data/{i}/{j}',f'test/{i}')
