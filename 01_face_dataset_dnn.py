import cv2
import os
import sys
DATA_SET = './dataset'

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# cam = cv2.VideoCapture('OUTPUT_FILE-2.avi')

modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# For each person, enter one numeric face id
face_id = int(input('\n Enter user ID end press <Enter> ==>  '))
print("\n Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
index = 0

ids = []
for f_name in os.listdir(DATA_SET):
    parts = f_name.split(".")
    if int(parts[1]) == face_id:
        ids.append(int(parts[2]))
ids.sort()

if len(ids):
    index = ids[-1]
print(f"start index:{index}")
conf_threshold=0.7

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame_height = img.shape[0]
    frame_width = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False,)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            try:
                print(f'detection i:{i} confidence:{confidence}')
                x = int(detections[0, 0, i, 3] * frame_width)
                y = int(detections[0, 0, i, 4] * frame_height)
                w = int(detections[0, 0, i, 5] * frame_width) - x
                h = int(detections[0, 0, i, 6] * frame_height) - y

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                index += 1
                count += 1
                # Save the captured image into the datasets folder
                f_name = f"{DATA_SET}/User.{face_id}.{index}.jpg"
                print(f_name)
                cv2.imwrite(f_name, gray[y:y+h,x:x+w])
            except cv2.error as e:
                print(f'error:{e}') 
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 300: # Take 30 face sample and stop video
        break
    cv2.imshow('image', img)

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
