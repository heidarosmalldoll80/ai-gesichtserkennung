from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Bild speichern
            img_path = './uploads/' + file.filename
            file.save(img_path)

            # Gesichtserkennung aufgerufen
            face_detection(img_path)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)


def face_detection(img_path):
    # Lade das Haar-Cascade-Klassifikator
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Bild laden und in Graustufen umwandeln
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Gesichter in das Bild zeichnen
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Speichern des bearbeiteten Bildes im statischen Verzeichnis
    cv2.imwrite('./static/result_' + img_path.split('/')[-1], img)


if __name__ == '__main__':
    app.run(debug=True)