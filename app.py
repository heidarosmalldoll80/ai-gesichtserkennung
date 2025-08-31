from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the image to the uploads directory
            img_path = './uploads/' + file.filename
            file.save(img_path)

            # Call face detection on the uploaded image
            face_detection(img_path)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    # Render the results page with the processed image filename
    return render_template('results.html', filename=filename)


def face_detection(img_path):
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image and convert it to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save the processed image in the static directory
    result_img_path = './static/result_' + img_path.split('/')[-1]
    cv2.imwrite(result_img_path, img)


if __name__ == '__main__':
    app.run(debug=True)