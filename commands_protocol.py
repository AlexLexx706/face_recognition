import re
import threading
from types import SimpleNamespace
import time
import ctypes
import serial


# bit 0 - BUTTON_0 ... bit 8 - BUTTON_8
# bit 9 - BUTTON_END_LEFT
# bit 10 - BUTTON_END_RIGHT
# bit 11-12 - state of finger
# bit 13 - state of door


class BitsButtons(ctypes.Structure):
    """Describe buttos states
    """
    _fields_ = (
        ("button_0", ctypes.c_uint16, 1),
        ("button_1", ctypes.c_uint16, 1),
        ("button_2", ctypes.c_uint16, 1),
        ("button_3", ctypes.c_uint16, 1),
        ("button_4", ctypes.c_uint16, 1),
        ("button_5", ctypes.c_uint16, 1),
        ("button_6", ctypes.c_uint16, 1),
        ("button_7", ctypes.c_uint16, 1),
        ("button_8", ctypes.c_uint16, 1),
        ("button_end_left", ctypes.c_uint16, 1),
        ("button_end_right", ctypes.c_uint16, 1),
        ("finger_state", ctypes.c_uint16, 2),
        ("door_state", ctypes.c_uint16, 1),
    )
    _pack_ = 1


class ButtonState(ctypes.Union):
    _fields_ = (
        ('bits', BitsButtons),
        ('value', ctypes.c_uint16)
    )


class BoxState(ctypes.Structure):
    """Describe box state
    """
    _fields_ = [
        # buttons states
        ("state", ButtonState),

        # position of finger 100..2500
        ("pos", ctypes.c_int16),

        # check sum (not used now)
        ("cs", ctypes.c_uint8),
    ]
    _pack_ = 1


class Protocol:
    """implementation of the protocol of interaction  between eyes controller and useless box controller
    """
    MAX_BUFFER_SIZE = 100

    def __init__(self):
        self._read_thread = None
        self._stop_event = threading.Event()
        self._serial = None
        self._cmd_index = 0
        self._re_header = re.compile(
            br'([A-Z]{2})([0-9A-F]{3})(?:%(\w*)%)?(.*)')
        self._state = ButtonState()
        self._cmd_condition = threading.Condition()
        self._last_re = SimpleNamespace(id='', prefix='', msg='')

    @property
    def state(self):
        return self._state

    def start(self, port: str = '/dev/ttyUSB0', speed: int = 115200):
        """start protocol
        Args:
            port (str, optional): file path of port. Defaults to '/dev/ttyUSB0'.
            speed (int, optional): port speed
        """
        if self._read_thread is not None:
            raise RuntimeError('protocol already running')
        self._cmd_index = 0
        self._serial = serial.Serial(port, speed, timeout=1.0)
        self._read_thread = threading.Thread(target=self._read_proc)
        self._stop_event.clear()
        self._read_thread.start()
        time.sleep(3)

    def stop(self):
        """stop protocol
        """
        if self._read_thread is None:
            raise RuntimeError('Reading serial thread not exist')
        self._stop_event.set()
        self._read_thread.join()
        self._serial.close()
        self._read_thread = None

    def _send_cmd(self, cmd: str):
        """send command and waiting responce

        Args:
            cmd (_type_): _description_
        """
        cur_prefix = self._cmd_index
        self._cmd_index += 1
        full_cmd = f'%{cur_prefix}%{cmd}'
        # debug
        print(full_cmd)
        self._serial.write(f'{full_cmd}\n'.encode())
        self._wait_responce(cur_prefix)

    def _wait_responce(self, prefix: str, timeout: int = 0.5):
        with self._cmd_condition:
            self._cmd_condition.wait(timeout)
            if self._last_re.prefix != prefix:
                raise RuntimeWarning(
                    f"prefix:{prefix}!= {self._last_re.prefix} in:{self._last_re.id} msg:{self._last_re.msg}")

            if self._last_re.id == 'ER':
                raise RuntimeError(
                    f"error:{self._last_re.msg}")

    def set_mode(self, mode: int):
        """set current mode

        Args:
            mode (int): 0 - auto mode; 1 - manual
        """
        self._send_cmd(f'set,/par/mode,{mode}')

    def open_door(self):
        """send cmd open door
        """
        self._send_cmd('set,/par/manual/door,1')

    def close_door(self):
        """send cmd open door
        """
        self._send_cmd('set,/par/manual/door,0')

    def activate_state_stream(self):
        """activate state stream
        """
        self._send_cmd('em,state')

    def stop_state_stream(self):
        """stop state stream
        """
        self._send_cmd('dm,state')

    def set_finger_state(self, state: int):
        """move finger servo
        Args:
            state (int):
                0 - init state
                1 - ready state
                2 - pres state
        """
        self._send_cmd(f'set,/par/manual/finger,{state}')

    def move_finger(self, pos):
        """move finger to pos

        Args:
            pos (int): position of finger: 100...2500
        """
        self._send_cmd(f'set,/par/manual/pos,{pos}')

    def _read_proc(self):
        """decode stream form serial port
        """
        data = b''
        while not self._stop_event.is_set():
            line = self._serial.readline()
            data += line
            # only for debug
            # print(repr(line))
            match = self._re_header.search(data)

            # header decoded
            if match:
                msg_id = match.group(1)
                msg_len = int(match.group(2), 16)
                start = match.end(2)
                cur_message_len = len(data) - start
                # all data present
                # print(f"msg_id:{msg_id} msg_len:{msg_len} cur_message_len:{cur_message_len}")

                if cur_message_len >= msg_len:
                    end = start + msg_len
                    # print("sasamba 1")
                    # commands responce
                    if msg_id in (b'RE', b'ER'):
                        # print("sasamba 2")
                        start = match.start(4)
                        msg_prefix = match.group(3)
                        # riase contition for waiter
                        with self._cmd_condition:
                            msg = data[start:end]
                            self._last_re.id = msg_id.decode()
                            self._last_re.prefix = int(msg_prefix)
                            self._last_re.msg = msg.decode()
                            self._cmd_condition.notify()
                            print(data[match.start(0):end].decode())
                    # status message
                    elif msg_id == b'BS':
                        # print("sasamba 3")
                        # update state
                        self._state = ButtonState.from_buffer_copy(
                            data[start:end])

                        # print("0:%d 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d 7:%d 8:%d e_l:%d e_r:%d f_s:%d d_s:%d" % (
                        #     self._state.bits.button_0,
                        #     self._state.bits.button_1,
                        #     self._state.bits.button_2,
                        #     self._state.bits.button_3,
                        #     self._state.bits.button_4,
                        #     self._state.bits.button_5,
                        #     self._state.bits.button_6,
                        #     self._state.bits.button_7,
                        #     self._state.bits.button_8,
                        #     self._state.bits.button_end_left,
                        #     self._state.bits.button_end_right,
                        #     self._state.bits.finger_state,
                        #     self._state.bits.door_state,
                        # ))
                    # cut data
                    data = data[end:]

            # cut unknown data
            if len(data) > self.MAX_BUFFER_SIZE:
                data = data[-self.MAX_BUFFER_SIZE:]


if __name__ == '__main__':
    protocol = Protocol()
    protocol.start(port='/dev/ttyUSB0')
    time.sleep(10)
    protocol.set_mode(1)
    protocol.activate_state_stream()
    time.sleep(1)
    protocol.open_door()
    time.sleep(1)
    protocol.close_door()
    time.sleep(1)
    protocol.set_finger_state(1)
    time.sleep(1)
    protocol.set_finger_state(2)
    time.sleep(1)
    protocol.set_finger_state(0)
    time.sleep(1)
    protocol.move_finger(100)
    time.sleep(1)
    protocol.move_finger(2000)
    print(f"state:{protocol.state}")
    time.sleep(2000)
    protocol.stop_state_stream()
    time.sleep(2)
    protocol.stop()
