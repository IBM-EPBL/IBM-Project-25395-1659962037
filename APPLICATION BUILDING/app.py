from flask import Flask,request, url_for, redirect, render_template
import tensorflow
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from skimage import io
from PIL import Image
import numpy as np

app = Flask(__name__)

model=load_model('mnist.h5')


@app.route('/')
def hello_world():
    return render_template("hand_written.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        
        img= Image.open(request.files['file'].stream).convert("1") 

        print(img)
        img=img.resize((28,28))
        im2arr = np.array(img)
        im2arr=im2arr.reshape(1,28,28,1)
        pred = model.predict(im2arr)
        num = np.argmax(pred)
        print(num)
        return render_template('hand_written.html', num=num)


if __name__ == '__main__':
    app.run(debug=True)