import cv2
import numpy as np
import os
import simpleaudio as sa
import multiprocessing
import queue
import ctypes
import time

SHOW_VIEW = True


def recognize_face(queue, recognition_state):
    wave_hi_alex = sa.WaveObject.from_wave_file("sounds/hi_alex.wav")
    wave_hi_agniia = sa.WaveObject.from_wave_file("sounds/hi_agniia.wav")
    wave_hi_anna = sa.WaveObject.from_wave_file("sounds/hi_anna.wav")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer/trainer.yml')
    names = ['Alex', 'Agniia', 'Anna']

    while 1:
        data = queue.get()
        if data is None:
            break
        gray, rect = data
        id, confidence = recognizer.predict(gray)
        name = "unknown"
        conf_str = "  0%"

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            name = names[id]
            conf_str = "  {0}%".format(round(100 - confidence))

            if name == 'Alex':
                wave_hi_alex.play().wait_done()
            elif name == 'Agniia':
                wave_hi_agniia.play().wait_done()
            elif name == 'Anna':
                wave_hi_anna.play().wait_done()

        #unlock recognition state
        with recognition_state.get_lock():
            recognition_state.value = 0

def main():
    image_queue = multiprocessing.Queue(1)
    recognition_state = multiprocessing.Value(ctypes.c_int, 0, lock=True)
    recognition_proc = multiprocessing.Process(
        target=recognize_face, args=(image_queue, recognition_state))
    recognition_proc.start()

    faceCascade = cv2.CascadeClassifier(
        "./haarcascade_frontalface_default.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    p_time = time.time()
    frames = 0

    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=1,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            print(f'face:{(x,y)}')

            if SHOW_VIEW:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            with recognition_state.get_lock():
                if not recognition_state.value:
                    recognition_state.value = 1
                    image_queue.put((gray[y:y+h, x:x+w], (x, y, w, h)))

            # name = "unknown"
            # conf_str = "  0%"

            # if SHOW_VIEW:
            #     cv2.putText(img, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
            #     cv2.putText(img, conf_str, (x+5, y+h-5),
            #                 font, 1, (255, 255, 0), 1)

        if SHOW_VIEW:
            cv2.imshow('camera', img)

        c_time = time.time()
        if c_time - p_time >= 1:
            print(f'fps:{frames}')
            p_time = c_time
            frames = 0
        frames += 1

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
