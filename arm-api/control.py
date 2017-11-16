#!/usr/bin/env python
import time
import serial
import threading
import sys
import os
import numpy as np
from denormalizer import Denormalizer

# usbport = '/dev/ttyACM2'
usbport = '/dev/cu.usbmodem1421'

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
        except:
            sys.exit("Error connecting device")

    def _readline(self):
        eol = b'\n'
        leneol = len(eol)
        line = bytearray()
        while True:
            c = self.dev.read(1)
            if c:
                line += c
                if line[-leneol:] == eol:
                    break
            else:
                break
        return bytes(line)

    def get_position(self, _, run_event):
        while run_event.is_set():
            try:
                pos = _readline()
                poses = pos.replace('\n', '').split(',')
                print(poses)
                if (len(poses) == 6):
                    self.feedback = poses
                    print("Position from arduino: ", self.feedback)
                pass
            except ValueError as ve:
                print(ve)
                pass

    def move(self, servo, angle):
        if (0 <= angle <= 180):
            self.dev.reset_input_buffer()
            self.dev.write(chr(255))
            self.dev.write(chr(servo))
            self.dev.write(chr(angle))
        else:
            print("Servo angle must be an integer between 0 and 180.\n")

    def set_positions(self, angles):
        denormalizer = Denormalizer(RANGES)
        denormalized = denormalizer.denormalize(angles)
        for idx, angle in enumerate(denormalized):
            time.sleep(0.1)
            print(angle)
            angle = int(angle)
            print(idx+1, angle)
            servo = idx + 1
            self.move(servo, angle)
            time.sleep(0.1)

if __name__ == '__main__':
    arm = Arm()

    positions = np.array([ -0.21467527, -0.26472031, 0.19183921, 0.17777404, 0.16154106, 0.18859741])

    # print(denormalized)
    arm.set_positions(positions)



