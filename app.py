import numpy as np
import cv2
import tensorflow as tf
import keras
from PIL import Image
import PIL
import pandas as pd
import keras.utils as image
from flask import *
import wikipedia
import os
import shutil
app=Flask(__name__)

def find(query):
    df=pd.read_excel('Result_Data.xlsx')
    print(query)
    inx=int(df.index[df['Plant Name']==query][0])
    print(inx)
    row=df[inx:inx+1]
    d0= row.values.tolist()[0][0]
    d1= row.values.tolist()[0][1]
    d2= row.values.tolist()[0][2]
    d3= row.values.tolist()[0][3]
    d4= row.values.tolist()[0][4]
    return d0,d1,d2,d3,d4
    
def Predict(image_path):
    classes=[
    'Betel Leaf',
    'Jackfruit',
    'Bilimbi',
    'Basil',
    'Lime',
    'Guava',
    'Celery',
    'Aloe Vera',
    'Screwpine',
    'Papaya'
]
    model=keras.models.load_model("Model.keras")
    img=image.load_img(image_path,target_size=(250,250,3))
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
        _,q1,q2,q3,q4=find(result)
        return result,q1,q2,q3,q4
    else:    
        return result,""


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
        os.remove(file.filename)
        return render_template("result.ejs",result=result,q1=q1,q2=q2,q3=q3,q4=q4)

if __name__=='__main__':
    app.run(debug=True)
    app.config['TEMPLATiES_AUTO_RELOAD']=True
