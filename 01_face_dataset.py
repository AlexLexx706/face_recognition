import cv2
import os
import sys
DATA_SET = './dataset'

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# cam = cv2.VideoCapture('OUTPUT_FILE-2.avi')

face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
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

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        index += 1
        count += 1
        # Save the captured image into the datasets folder
        f_name = f"{DATA_SET}/User.{face_id}.{index}.jpg"
        print(f_name)
        cv2.imwrite(f_name, gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
