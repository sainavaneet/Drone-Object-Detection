import csv
import cv2
import math
import os
import shlex
import subprocess
import tempfile
import numpy as np
import olympe
import olympe_deps as od
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, AltitudeChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
import time
import sys
import math
import collections


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = 'rtsp_transport;udp'
TIMEOUT = 120
DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")
DRONE_IP = "192.168.42.1"
olympe.log.update_config({"loggers": {"olympe": {"level": "INFO"}}})
drone = olympe.Drone(DRONE_IP)
#drone.connect(retry=3)


def fly(drone):


    drone(
        FlyingStateChanged(state="hovering", _policy="check")
        | FlyingStateChanged(state="flying", _policy="check")
        | (
            GPSFixStateChanged(fixed=1, _timeout=TIMEOUT, _policy="check_wait")
            >> (
                TakeOff(_no_expect=True)
                & FlyingStateChanged(
                    state="hovering", _timeout=TIMEOUT, _policy="check_wait"
                )
            )
        )
    ).wait()


fly(drone)



cap = cv2.VideoCapture('5.mp4')
#cap = cv2.VideoCapture('rtsp://192.168.42.1/live', cv2.CAP_FFMPEG)
net = cv2.dnn.readNetFromONNX("best.onnx")
file = open("coco.txt","r")
classes = file.read().split('\n')
print(classes)

while True:
    
    vx=0.0;
    vy=0.0;
    vz=0.0;
    vr=0.0;
    vxx=0.0;
    vyy=0.0;
    vzz=0.0;
    marker_y=0.0;
    marker_x=0.0;
    kp=0.005;
    global n;
    global count;
    drone.piloting_pcmd(0, 0, 0, 0, 0.0)
    time.sleep(1)
    img = cap.read()[1]
    if img is None:
        break
    img = cv2.resize(img, (640,600))
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
  

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
    cv2.circle(img,(int(img.shape[1]/2),int(img.shape[0]/2)),2,(0,0,255),-1) 

    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        c_x = x1 + w // 2
        c_y = y1 + h // 2
        cv2.circle(img, (c_x, c_y), 2, (0, 0, 255), -1)
        vy=0.0
        vx=0.1
        vz=kp*(img.shape[0]/2-c_y)
        vr=kp*(img.shape[1]/2-c_x)
        r11=pose[0, 0]
        r21=pose[1, 0]
        r31=pose[2, 0]
        r32=pose[2, 1]
        r33=pose[2, 2]
        yaw=math.atan(r21/r11)
        pitch=math.atan(-r31/math.sqrt(math.pow(r32,2)+math.pow(r33,2)))
        roll=math.atan(r32/r33)

        vxx=int(roll*200)
        vyy=int(pitch*100)
        vzz=int(yaw*400)
        print(vxx,vyy,vzz)
        distance=pose[2, 3]
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(img, text, (x1,y1-3),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,0,255),2)
        print(vx , vy , -vz, -vr)
        #drone(moveBy(vx, vy, -vz, -vr)).wait(30)
        drone.piloting_pcmd(vxx, 0, vzz, 0, 1)

    cv2.imshow("VIDEO",img)
    k = cv2.waitKey(120)
    if k == ord('q'):
        break
