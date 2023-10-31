import os
import time
import queue
import threading
import simpleaudio as sa
import logging
import commands_protocol

LOG = logging.getLogger(__name__)


class BoxProcessor:
    """Processed box states
    """
    BASE_PATH = ''
    SLEEP_BEFORE_PLAY = 0.5

    def __init__(self) -> None:
        self._buttons_sounds = {
            i: sa.WaveObject.from_wave_file(
                os.path.join(self.BASE_PATH, f"sounds/buttons/{i + 1}.wav")) for i in range(9)}

        self._protocol = commands_protocol.Protocol()
        self._protocol.start(port='/dev/ttyUSB0')
        self._protocol.activate_state_stream()

        self._buttons_queue = queue.Queue()
        self._bottons_speaker_thread = threading.Thread(
            target=self._bottons_speaker, args=(self._buttons_queue,))
        self._bottons_speaker_thread.start()

    def _bottons_speaker(self, buttons_queue):
        """play buttons sounds

        Args:
            buttons_queue (Queue): queue used for receive pressed buttons
        """
        last_time = time.time()
        while 1:
            num = buttons_queue.get()
            LOG.debug('play btn:%s', num)

            # sleep if no buttons pressed before
            if (time.time() - last_time) > self.SLEEP_BEFORE_PLAY:
                time.sleep(self.SLEEP_BEFORE_PLAY)

            self._buttons_sounds[num].play().wait_done()
            last_time = time.time()

    def process(self):
        """checks buttons states changed
        """
        prev_buttons_state = self._protocol.state.value & 0b111111111

        # checks buttons
        while 1:
            cur_buttons_state = self._protocol.state.value & 0b111111111
            # buttons pressed
            if cur_buttons_state != prev_buttons_state:
                for i in range(9):
                    mask = 1 << i
                    if cur_buttons_state & mask and (not prev_buttons_state & mask):
                        self._buttons_queue.put(i)

            prev_buttons_state = cur_buttons_state
            time.sleep(0.1)
