import cv2
import numpy as np
import glob
import random


# Load Yolo
net = cv2.dnn.readNet("weights/yolo-drone.weights", "cfg/yolo-drone.cfg")

# Name custom object
classes = ["drone"]

# Images path
cap = cv2.VideoCapture(0)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    ret, vid = cap.read()

    vid = cv2.resize(vid, (0,0), fx=0.6, fy=0.6)
    height, width, channels = vid.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(vid, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
            if confidence > 0.3:
                # Object detected
                
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                
                

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    cv2.circle(vid,(int(vid.shape[1]/2),int(vid.shape[0]/2)),4,(0,0,255),-1)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(vid, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vid, label, (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.circle(vid, (cx, cy), 2, (0, 0, 255), -1)
            
            

    cv2.imshow('live feed', vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
