import time
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
import logging

LOG = logging.getLogger(__name__)


class EyesProcessor:
    TOP_BUTTON = 23
    BOTTOM_BUTTON = 22

    FREQUENCE = 20  # frequence of main loop
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
