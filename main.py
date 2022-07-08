from flask import Flask, render_template, request, session
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import torch
from model import create_model

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__, template_folder='templateFiles',
            static_folder='staticFiles')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'Password'

CLASSES = ['Flute', 'Bed', 'Bridge', 'Broom', 'Door', 'Fencing', 'Grain Storage', 'Hand Fan', 'House', 'Mask', 'Jakoi', 'Khaloi',
           'Dhol', 'Pepa', 'Japi', 'Gogona', 'Winnowing Fan', 'Toka', 'Julki']

# CLASSES = ['Flute',
#            'Jakoi',
#            'Khaloi',
#            'ban bati',
#            'Bell Metal Bota',
#            'Bell Metal Spoon',
#            'bell metal bowl',
#            'Bell Metal Plate',
#            'Ban Bati',
#            'Bell Metal Glass',
#            'bell metal bota',
#            'bell metel cymbal',
#            'Jaw harp',
#            'Dhol',
#            'Pepa',
#            'Toka',
#            'Julki',
#            'Conical Hat']

NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('Axom2.pth', map_location=DEVICE))
model.eval()

detection_threshold = 0.8


def detect_object(uploaded_image_path):
    image = cv2.imread(uploaded_image_path)
    orig_image = image.copy()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)
        output_image_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'output_image.jpg')
        cv2.imwrite(output_image_path, orig_image)
        return(output_image_path)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(
            app.config['UPLOAD_FOLDER'], img_filename))

        session['uploaded_img_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('home.html')


@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('image.html', user_image=img_file_path)


@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    return render_template('image.html', user_image=output_image_path)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
