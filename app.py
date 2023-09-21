import numpy as np
import cv2
import tensorflow as tf
import keras
from PIL import Image
import PIL
import numpy as np
import keras.utils as image
from flask import *
import wikipedia
import os
app=Flask(__name__)

def find(query):
    print(query)
    web_data=wikipedia.summary(query,auto_suggest=False,sentences=13)
    return web_data
def Predict(image_path):
    classes=[
    "Papaya",
    "Jackfruit",
    "Lemon Basil",
    "Lime",
    "Aloe Vera",
    "Guava",
    "Betel Leaf",
    "Celery",
    "Bilimbi",
    "Screwpine"]
    model=keras.models.load_model("Herbs_classifier.keras")
    img=image.load_img(image_path,target_size=(150,150,3))
    img=image.img_to_array(img)
    test_image = np.expand_dims(img, axis = 0)
    result = model.predict(test_image)
    ind=0
    if 1.0 in result[0]:
        for i in result[0]:
            if int(i)==1:
                result=str(classes[ind])
                break
            else:
    
                ind+=1
    else:
        result='unknown'
    if result != 'unknown':
        info=find(result)
        return result,info
    else:    
        return result,''

@app.route('/')
def home():
    return render_template("index.ejs")
@app.route('/camera')
def Cam():
    return render_template('camera.ejs')
@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        file=request.files['file']
        file.save(file.filename)
        result,q1,q2,q3,q4=Predict(file.filename)
        shutil.copy(f'{file.filename}','static/input.jpg')
        print(file.filename)
        return render_template("result.ejs",result=result,q1=q1,q2=q2,q3=q3,q4=q4)
if __name__=='__main__':
    app.run(debug=True)
    app.config['TEMPLATiES_AUTO_RELOAD']=True
