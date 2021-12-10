import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
#Brain/

model =load_model('D:\Computer Vision\Brain\BrainTumor10EpochsCategorical.h5')
modelm =load_model('D:\Computer Vision\Brain\Malaria.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predict_x=model.predict(input_img) 
    result=np.argmax(predict_x,axis=1)
    return result


@app.route('/', methods=['GET'])
def Brain():
    return render_template('mainpg.html')
@app.route('/Brain', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/Malaria', methods=['GET'])
def indexm():
    return render_template('MalariaIndex.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' :
        if 'test' not in request.form:
            f = request.files['file']

            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            value=getResult(file_path)
            result=get_className(value) 
            return result
        else :
            f = request.files['file']

            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            value=getResultm(file_path)
            resultm=get_classNamem(value) 
            return resultm

    return None

###
def get_classNamem(classNo):
	if classNo==1:
		return "No Malaria"
	elif classNo==0:
		return "Yes Malaria"


def getResultm(imgr):
    # image=cv2.imread(img)
    # image = Image.fromarray(image, 'RGB')
    # image = image.resize((64, 64))
    # image=np.array(image)
    # input_img = np.expand_dims(image, axis=0)
    # predict_x=modelm.predict(input_img) 
    # result=np.argmax(predict_x,axis=1)
    # return result
    img=image.load_img(imgr,target_size=(64,64) )
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=modelm.predict(images)

    return val
                                                    
@app.route('/predictm', methods=['GET', 'POST'])
def uploadm():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResultm(file_path)
        resultm=get_classNamem(value) 
        return resultm
    return None


if __name__ == '__main__':
    app.run(debug=True)