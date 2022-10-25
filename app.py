import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Defint the flask app
app = Flask(__name__)

# Load Train Model
MODEL_PATH = 'resnet50_model.h5'

model = load_model(MODEL_PATH)


# Create a function
def model_predict(img_path, model):
    img = image.load_model(img_path, target_size=(256,256))
    x = image.img_to_array(img)
    x = x/255
    x = np.exapan_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        print('A')
    elif preds == 1:
        print('B')
    elif preds == 2:
        print('C')
    elif preds == 3:
        print('D')
    elif preds == 4:
        print('E')
    elif preds == 5:
        print('F')
    elif preds == 6:
        print('G')
    elif preds == 7:
        print('H')
    elif preds == 8:
        print('I')
    elif preds == 9:
        print('K')
    elif preds == 10:
        print('L')
    elif preds == 11:
        print('M')
    elif preds == 12:
        print('N')
    elif preds == 13:
        print('O')
    elif preds == 14:
        print('P')
    elif preds == 15:
        print('Q')
    elif preds == 16:
        print('R')
    elif preds == 17:
        print('S')
    elif preds == 18:
        print('T')
    elif preds == 19:
        print('U')
    elif preds == 20:
        print('V')
    elif preds == 21:
        print('W')
    elif preds == 22:
        print('X')
    else:
        print('Y')

    return preds

@app.route('/' ,  methods =['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def uploads():
    if request.method == 'POST':
        f = request.files['file']

        # Save files to uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)






