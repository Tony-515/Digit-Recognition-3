from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from keras import models
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Open the model
model = models.load_model('digit_recognition.keras')

def load_picture(path):
    with Image.open(path).convert('RGB').resize((32, 32)) as img:
        return img

def process(img):
    print(img)
    img = img.convert('L').resize((28, 28))
    img = ImageOps.invert(img) if img.getpixel((0, 0)) > 127 else img
    img = np.array(img)
    img = img.reshape((28, 28, 1)).astype('float32') / 255
    return img

# Make a prediction with a pre-processed image
def predict(model, img):
    img = process(img)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return labels[model.predict(img[None, :, :]).argmax()]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(url_for('index'))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            prediction = predict(model, load_picture(filepath))
            return render_template('index.html', result=f"{filename} looks like a(n) {prediction}!", filepath=filepath)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='localhost', port='5000', debug=True)