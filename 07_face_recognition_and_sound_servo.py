import os
import cv2
import simpleaudio as sa
import multiprocessing
import ctypes
import time
import commands_protocol
import logging
from types import SimpleNamespace
import queue
import threading
from processors import eyes_processor
from processors.box_processor import BoxProcessor

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SHOW_VIEW = False
USE_DNN = True

BASE_PATH = os.path.split(__file__)[0]

IMAGES_PATH = "./images"

def create_directory(directory):
   try:
      os.makedirs(directory)
      LOG.debug(f"Directory '{directory}' created successfully")
   except OSError as e:
      LOG.exception(f"Error: {e}")



def recognize_face(queue, recognition_state):
    wave_hi_alex = sa.WaveObject.from_wave_file(
        os.path.join(BASE_PATH, "sounds/hi_alex.wav"))
    wave_hi_agniia = sa.WaveObject.from_wave_file(
        os.path.join(BASE_PATH, "sounds/hi_agniia.wav"))
    wave_hi_anna = sa.WaveObject.from_wave_file(
        os.path.join(BASE_PATH, "sounds/hi_anna.wav"))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(BASE_PATH, 'trainer/trainer.yml'))
    names = ['Alex', 'Agniia', 'Anna']

    create_directory(IMAGES_PATH)

    while 1:
        data = queue.get()
        if data is None:
            break
        gray, rect = data
        id, confidence = recognizer.predict(gray)
        name = "unknown"
        conf_str = "  0%"

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence > 80):
            name = names[id]
            conf_str = "  {0}%".format(round(100 - confidence))

            if name == 'Alex':
                wave_hi_alex.play().wait_done()
            elif name == 'Agniia':
                wave_hi_agniia.play().wait_done()
            elif name == 'Anna':
                wave_hi_anna.play().wait_done()

            if name in names:
                img_path = os.path.join(IMAGES_PATH, "last_recognition.jpg")
                LOG.debug(f"name:{name} names:{names} confidence:{confidence} img_path:{img_path}")
                cv2.imwrite(img_path, gray)

        time.sleep(3)
        # unlock recognition state
        with recognition_state.get_lock():
            recognition_state.value = 0


def find_face_dnn(img, net, conf_threshold):
    frame_height = img.shape[0]
    frame_width = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
        104, 117, 123], False, False,)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x = int(detections[0, 0, i, 3] * frame_width)
            y = int(detections[0, 0, i, 4] * frame_height)
            w = int(detections[0, 0, i, 5] * frame_width) - x
            h = int(detections[0, 0, i, 6] * frame_height) - y
            yield (x, y, w, h)


def process_servo(face_pos: object):
    """Process servo movements
    Args:
        face_pos (list): position of face (x, y)
    """
    processor = eyes_processor.EyesProcessor()
    processor.process(face_pos)


def process_box():
    """process box states
    """
    BoxProcessor.BASE_PATH = BASE_PATH
    processor = BoxProcessor()
    processor.process()


def main():
    image_queue = multiprocessing.Queue(1)
    recognition_state = multiprocessing.Value(ctypes.c_int, 0, lock=True)

    recognition_proc = multiprocessing.Process(
        target=recognize_face, args=(image_queue, recognition_state))
    recognition_proc.start()

    box_proc = multiprocessing.Process(
        target=process_box)
    box_proc.start()

    face_pos = multiprocessing.Value(
        ctypes.c_int * 2, lock=True)
    face_pos[0] = int(640 / 2)
    face_pos[1] = int(480 / 2)

    servo_proc = multiprocessing.Process(
        target=process_servo, args=(face_pos, ))
    servo_proc.start()

    if not USE_DNN:
        faceCascade = cv2.CascadeClassifier(os.path.join(
            BASE_PATH, "haarcascade_frontalface_default.xml"))
    else:
        modelFile = os.path.join(
            BASE_PATH, "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        configFile = os.path.join(BASE_PATH, "models/deploy.prototxt")
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        conf_threshold = 0.7

    cam_fps = 10

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set video widht
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set video height
    cam.set(cv2.CAP_PROP_FPS, cam_fps)

    past_fps_time = time.time()
    frames = 0
    process_timeout = 1./cam_fps
    past_process_time = past_fps_time
    readed_frames = 0
    show_fps = False

    while True:
        _, img = cam.read()
        readed_frames += 1
        cur_time = time.time()

        duration = cur_time - past_process_time
        if duration >= process_timeout:
            past_process_time = cur_time - (duration % process_timeout)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if not USE_DNN:
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=1,
                    minSize=(100, 100))
            else:
                faces = find_face_dnn(img, net, conf_threshold)

            for (x, y, w, h) in faces:
                with face_pos.get_lock():
                    face_pos[0] = x
                    face_pos[1] = y

                if SHOW_VIEW:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # send data to recognition
                with recognition_state.get_lock():
                    if not recognition_state.value:
                        recognition_state.value = 1
                        image_queue.put((gray[y:y+h, x:x+w], (x, y, w, h)))

            if SHOW_VIEW:
                cv2.imshow('camera', img)

            if show_fps:
                frames += 1

            if SHOW_VIEW:
                k = cv2.waitKey(1) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break

        if show_fps and cur_time - past_fps_time >= 1:
            LOG.debug('fps:%s reads:%s', frames, readed_frames)
            past_fps_time = cur_time
            frames = 0
            readed_frames = 0

    # Do a bit of cleanup
    LOG.debug("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
