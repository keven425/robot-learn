#!/usr/bin/env python
import time
import serial
import threading
import sys
import os
import numpy as np
from arm.denormalizer import Denormalizer

# usbport = '/dev/ttyACM2'
# usbport = '/dev/cu.usbmodem1421'

# Set up serial baud rate
# ser = serial.Serial(usbport, 9600, timeout=1)

# min, max, home
BOUNDS = {
    1: [144, 496, 331],
    2: [222, 468, 303],
    3: [375, 319, 156],
    4: [143, 475, 308],
    5: [130, 471, 318],
    6: [133, 440, 289],
}

RANGES = [
    [7, 180],
    [46, 169],
    [122, 94],
    [6, 172],
    [0, 170],
    [1, 155]
]
# 0
# ['144', '222', '375', '143', '130', '133']
# 90
# ['331', '303', '319', '308', '318', '289']
# 180
# ['496', '468', '156', '475', '471', '440']
# 160 '457' / 20 '176'



class Arm(object):
    def __init__(self):
        self.port="/dev/cu.usbmodem1421"
        self.feedback = []
        try:
            self.dev = serial.Serial(self.port, 9600, timeout=5, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
            for i in range(100):
                self.dev.write(bytes([0]))
                time.sleep(0.03)
            self.dev.write(bytes([254]))
            print('wrote 254')
        except:
            sys.exit("Error connecting device")

    def _readline(self):
        eol = b'\n'
        leneol = len(eol)
        line = bytearray()
        while True:
            c = self.dev.read(1)
            print('c: ' + str(c))
            if c:
                line += c
                if line[-leneol:] == eol:
                    break
            else:
                break
        return bytes(line)

    # def get_position(self, _, run_event):
    def get_positions(self):
        self.feedback = []
        try:
            pos = self._readline()
            print(pos)
            poses = pos.replace('\n', '').split(',')
            print(poses)
            if (len(poses) == 6):
                self.feedback = poses
                print("Position from arduino: ", self.feedback)
                return self.feedback
            pass
        except ValueError as ve:
            print(ve)
            pass

    def move(self, servo, angle):
        if (0 <= angle <= 180):
            print(bytes([servo]))
            print(bytes([angle]))
            self.dev.reset_input_buffer()
            self.dev.write(bytes([255]))
            self.dev.write(bytes([servo]))
            self.dev.write(bytes([angle]))
        else:
            print("Servo angle must be an integer between 0 and 180.\n")

    def move_all(self, angles):
        # time.sleep(0.01)
        self.dev.reset_input_buffer()
        self.dev.write(bytes([255]))
        assert(len(angles) == 6)
        assert(np.all(angles <= 180))
        assert(np.all(angles >= 0))

        for i, angle in enumerate(angles):
            angle = int(angle)
            print('servo %d angle %d' % (i, angle))
            self.dev.write(bytes([angle]))

    def set_positions(self, angles):
        denormalizer = Denormalizer(RANGES)
        denormalized = denormalizer.denormalize(angles)
        time.sleep(0.01)
        self.move_all(denormalized)
        # while len(self.feedback) != 6:
        #     self.get_position()
        #     pass
        # return self.feedback
        # for idx, angle in enumerate(denormalized):
        #     angle = int(angle)
        #     print(idx+1, angle)
        #     servo = idx + 1
        #     self.move(servo, angle)

    # def home(self):
    #     time.sleep(0.1)
    #     self.move(1,30)
    #     time.sleep(0.03)
    #     self.move(1,120)
    #     time.sleep(0.03)
    #     self.move(1,30)
    #     time.sleep(0.03)
    #     time.sleep(0.5)
    #     self.move(1,120)
    #     # time.sleep(0.5)
    #     self.move(2,70)
    #     # time.sleep(0.5)
    #     self.move(3,70)
    #     # time.sleep(0.5)
    #     self.move(4,70)
    #     # time.sleep(0.5)
    #     self.move(5,70)
    #     # time.sleep(0.5)
    #     self.move(6,70)
    #     time.sleep(0.5)

# if __name__ == '__main__':
#     arm = Arm()

#     positions = np.array([ -0.11467527, -0.16472031, 0.19183921, 0.17777404, 0.16154106, 0.18859741])

#     # print(denormalized)
#     arm.set_positions(positions)



