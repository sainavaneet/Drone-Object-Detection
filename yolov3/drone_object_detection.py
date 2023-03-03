
#importing libraries------------------------------------------------------------
import cv2
import numpy as np
import os
import tempfile
import olympe
from imutils.video import VideoStream
from imutils.video import FPS
from pynput.keyboard import Listener, Key, KeyCode
import numpy as np
import subprocess
import argparse
import imutils
import time
import cv2
import os
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy,PCMD

class start_drone:
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(
           "10.202.0.1")        
    def start(self):
        
        self.drone.connect(retry=3)
        self.drone.start_piloting()
        

        self.drone(
                FlyingStateChanged(state="hovering", _policy="check")
        |          FlyingStateChanged(state="flying", _policy="check")
        |          (
                    GPSFixStateChanged(fixed=1, _timeout=TIMEOUT, _policy="check_wait")
                    >> (
                        TakeOff(_no_expect=True)
                        & FlyingStateChanged(
                        state="hovering", _timeout=TIMEOUT, _policy="check_wait"
                        )
                    )
                )
            ).wait()

        self.drone.piloting_pcmd(0, 0, 0, 0, 0.0)
        net = cv2.dnn.readNet("path to weights file", "path to .cfg file")
        self.layer_names = net.getLayerNames()
        classes = ["Drone"]
        cap = cv2.VideoCapture('5.mp4') # cv2.VideoCapture(0) for live stream


        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        while True:
            ret, img = cap.read()
            vx=0.0;
            vy=0.0;
            vz=0.0;
            vr=0.0;
            kp=0.002;

            img = cv2.resize(img, (0,0), fx=0.2,fy =0.2)
            height, width, channels = img.shape

    # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(outputlayers)

    # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
	    
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.circle(img,(int(img.shape[1]/2),int(img.shape[0]/2)),2,(0,0,255),-1) 
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    c_x = x + w // 2
                    c_y = y + h // 2
                    cv2.circle(img, (c_x, c_y), 2, (0, 0, 255), -1)
                    vy=0.0
                    vx=0.1
                    vz=kp*(img.shape[0]/2-c_y)
                    vr=kp*(img.shape[1]/2-c_x)
            
                    cv2.putText(img, label, (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
                       
                    
    
            
            cv2.imshow('live feed', img)
            self.drone(moveBy(vx, vy, -vz, -vr)).wait(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
if __name__ == "__main__":
    TIMEOUT = 120

     
    n   = start_drone()
    n.start()
