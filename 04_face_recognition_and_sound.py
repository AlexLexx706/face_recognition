import cv2
import numpy as np
import os
import simpleaudio as sa
import multiprocessing
import queue
import ctypes
import time

def play_sound(queue, sound_play):
    wave_hi_alex = sa.WaveObject.from_wave_file("sounds/hi_alex.wav")
    wave_hi_agniia = sa.WaveObject.from_wave_file("sounds/hi_agniia.wav")
    wave_hi_anna = sa.WaveObject.from_wave_file("sounds/hi_anna.wav")

    while 1:
        name = queue.get()
        if name is None:
            break
        if name == 'Alex':
            wave_hi_alex.play().wait_done()
        elif name == 'Agniia':
            wave_hi_agniia.play().wait_done()
        elif name == 'Anna':
            wave_hi_anna.play().wait_done()

        time.sleep(5)
        with sound_play.get_lock():
            sound_play.value = 0
        


def main():
    sound_queue = multiprocessing.Queue(1)
    sound_play = multiprocessing.Value(ctypes.c_int, 0, lock=True)
    sound_proc = multiprocessing.Process(target=play_sound, args=(sound_queue, sound_play))
    sound_proc.start()


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer/trainer.yml')
    cascadePath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # names related to ids: example ==> KUNAL: id=1,  etc
    names = ['Alex', 'Agniia', 'Anna']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=1,
            minSize=(100,100)
        )

        for (x, y, w, h) in faces:
            print(f'face:{(x,y)}')
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            id, confidence = 0, 50#recognizer.predict(gray[y:y+h, x:x+w])
            name = "unknown"
            conf_str = "  0%"

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                name = names[id]
                conf_str = "  {0}%".format(round(100 - confidence))

                with sound_play.get_lock():
                    if sound_play.value == 0:
                        sound_play.value = 1
                        sound_queue.put(name)

            cv2.putText(img, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, conf_str, (x+5, y+h-5),
                        font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
