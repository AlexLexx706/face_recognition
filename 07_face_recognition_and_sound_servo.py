import os
import cv2
import simpleaudio as sa
import multiprocessing
import ctypes
import time
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
import commands_protocol
import logging
from types import SimpleNamespace


LOG = logging.getLogger(__name__)


SHOW_VIEW = False
USE_DNN = True

BASE_PATH = os.path.split(__file__)[0]


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


class EyesProcessor:
    TOP_BUTTON = 23
    BOTTOM_BUTTON = 22

    FREQUENCE = 30  # frequence of main loop
    TAU = 0.2  # control speed of servo

    SHOW_FPS = False

    class ServoAngle:
        """helper for power off servo when not used
        """

        def __init__(self, servos_kit, num) -> None:
            self._prev_angle = None
            self._cnt = 0
            self._servos_kit = servos_kit
            self._num = num

        def set_angle(self, angle: int):
            """set angle to servo or disable it

            Args:
                angle (int): angle
            """
            if self._prev_angle == angle:
                # freeze servo
                if self._cnt > EyesProcessor.FREQUENCE:
                    # print(f"power off num:{self._num}")
                    self._servos_kit._pca.channels[self._num].duty_cycle = 0
                else:
                    self._cnt += 1
            else:
                self._cnt = 0
                try:
                    self._servos_kit.servo[self._num].angle = angle
                    # print(f"set angle:{angle} num:{self._num}")
                except ValueError:
                    LOG.warning(
                        "angle:%d out of range: 0..%d for servo:%d",
                        angle, self._servos_kit.servo[self._num].actuation_range, self._num)
            self._prev_angle = angle

    def __init__(self) -> None:
        self._servos_kit = ServoKit(channels=16)
        self._prepare_servo()
        self._v_angle = self.ServoAngle(self._servos_kit, 0)
        self._h_angle = self.ServoAngle(self._servos_kit, 1)

    def _prepare_servo(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TOP_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.BOTTOM_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self._servos_kit.continuous_servo[2].throttle = 0.3
        self._servos_kit._pca.channels[4].duty_cycle = 0xff
        self._servos_kit._pca.channels[5].duty_cycle = 0x6ff

        try:
            while 1:
                top_btn_state = GPIO.input(self.TOP_BUTTON)
                if top_btn_state:
                    self._servos_kit.continuous_servo[2].throttle = 0
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            self._servos_kit.continuous_servo[2].throttle = 0
            GPIO.cleanup()

    def _move_servo(self, pos):
        k_x = (640 - pos[0]) / 640
        k_y = (pos[1]) / 480
        self._v_angle.set_angle(int((180 - 30) * k_x + 30))
        self._h_angle.set_angle(int((180 - 30) * k_y + 30))

    def process(self, face_pos: object):
        """Control servo motors

        Args:
            face_pos (list): position of face
        """
        period = 1. / self.FREQUENCE
        past_time = time.time()
        fps_start_time = past_time
        count = 0
        cur_pos = [640./2., 480. / 2.]

        while 1:
            cur_time = time.time()
            duration = cur_time - past_time

            # make all operation at 30 HZ:
            if duration >= period:
                past_time = cur_time - duration % period
                with face_pos.get_lock():
                    pos = (face_pos[0], face_pos[1])

                cur_pos[0] += (pos[0] - cur_pos[0]) * duration / self.TAU
                cur_pos[1] += (pos[1] - cur_pos[1]) * duration / self.TAU
                self._move_servo(cur_pos)

                count += 1
            else:
                time.sleep(period - duration)

            # show servo fps
            if self.SHOW_FPS:
                duration = cur_time - fps_start_time
                if duration >= 1.:
                    LOG.debug('servo_fps:%s', count)
                    count = 0
                    fps_start_time = cur_time - duration % 1.


def process_servo(face_pos: object):
    """Process servo movements
    Args:
        face_pos (list): position of face (x, y)
    """
    processor = EyesProcessor()
    processor.process(face_pos)


def process_box():
    buttons_sounds = {
        i: sa.WaveObject.from_wave_file(
            os.path.join(BASE_PATH, f"sounds/buttons/{i + 1}.wav")) for i in range(9)}

    # create protocol object and activate buttons states
    protocol = commands_protocol.Protocol()
    protocol.start(port='/dev/ttyUSB0')
    time.sleep(10)
    protocol.activate_state_stream()
    time.sleep(1)

    prev_buttons_state = protocol.state.value & 0b111111111
    # checks buttons
    while 1:
        cur_buttons_state = protocol.state.value & 0b111111111

        # buttons pressed
        if cur_buttons_state != prev_buttons_state:
            for i in range(9):
                mask = 1 << i
                if cur_buttons_state & mask and (not prev_buttons_state & mask):
                    LOG.debug('play btn:%s', i)
                    buttons_sounds[i].play()

        prev_buttons_state = cur_buttons_state
        time.sleep(0.1)


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
