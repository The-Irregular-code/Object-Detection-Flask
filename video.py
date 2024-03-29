import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
from model import create_model

CLASSES = ['Flute', 'Bed', 'Bridge', 'Broom', 'Door', 'Fencing', 'Grain Storage', 'Hand Fan', 'House', 'Mask', 'Jakoi', 'Khaloi',
           'Dhol', 'Pepa', 'Conical Hat', 'Jaw Harp', 'Winnowing Fan', 'Toka', 'Julki']

NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='video.mp4'
)
args = vars(parser.parse_args())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


model = create_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
model.eval()

detection_threshold = 0.8
RESIZE_TO = (720, 540)
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = str(pathlib.Path(args['input'])).split(
    os.path.sep)[-1].split('.')[0]
out = cv2.VideoWriter(f"output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      RESIZE_TO)
frame_count = 0
total_fps = 0
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, RESIZE_TO)
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)  # .cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i]
                            for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, 2)
                cv2.putText(frame, class_name,
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)
        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
