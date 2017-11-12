#!/usr/bin/env python
import time
import serial
import threading
import sys
import os

usbport = '/dev/ttyACM2'
feedback = []

# Set up serial baud rate
ser = serial.Serial(usbport, 9600, timeout=1)

def _readline():
    eol = b'\n'
    leneol = len(eol)
    line = bytearray()
    while True:
        c = ser.read(1)
        if c:
            line += c
            if line[-leneol:] == eol:
                break
        else:
            break
    return bytes(line)

def move(servo, angle):
    '''Moves the specified servo to the supplied angle.

    Arguments:
        servo
          the servo number to command, an integer from 1-4
        angle
          the desired servo angle, an integer from 0 to 180

    (e.g.) >>> servo.move(2, 90)
           ... # "move servo #2 to 90 degrees"'''

    if (0 <= angle <= 180):
        ser.reset_input_buffer()
        ser.write(chr(255))
        ser.write(chr(servo))
        ser.write(chr(angle))
        # while True:
        #     try:
        #         time.sleep(0.01)
        #         state = _readline()
        #         print state
        #         return state
        #     except:
        #         pass
        # time.sleep(0.1)
    else:
        print("Servo angle must be an integer between 0 and 180.\n")

def init():

    # time.sleep(0.03)
    # move(1,30)
    # time.sleep(0.03)
    # move(1,120)
    # time.sleep(0.03)
    # move(1,30)
    # time.sleep(0.03)
    # 0
    # ['144', '222', '375', '143', '130', '133']
    # 90
    # ['331', '303', '319', '308', '318', '289']
    # 180
    # ['496', '468', '156', '475', '471', '440']

    # 160 '457' / 20 '176'

    time.sleep(1)
    move(1,90)
    time.sleep(1)
    move(2,90)
    time.sleep(1)
    move(3,20)
    time.sleep(1)
    move(4,90)
    time.sleep(1)
    move(5,90)
    time.sleep(1)
    move(6,90)
    time.sleep(1)

def Task1(_, run_event):

    while run_event.is_set():
        try:
            # pos = ser.read(1)
            # pos = ser.readline()
            # read = ser.read(1)
            # b1 = ord(read)
            # b2 = ord(read)
            # pos = b1 + b2*256
            pos = _readline()
            poses = pos.replace('\n', '').split(',')
            print poses
            if (len(poses) == 6):
                feedback = poses
                print "Position from arduino: ", feedback
            pass
        except ValueError as ve:
            print ve
            pass

def Task2():
    print "Inside Thread 2"
    init()

def Main():
    run_event = threading.Event()
    run_event.set()

    d1 = 1
    t1 = threading.Thread(target = Task1, args = ("task", run_event))
    d2 = 1
    t2 = threading.Thread(target = Task2)
    print "Starting Thread 1"
    t1.start()
    print "Starting Thread 2"
    t2.start()

    print "=== exiting ==="

    # init()
    try:
        while 1:
            time.sleep(.1)
    except KeyboardInterrupt:
        print "attempting to close threads. Max wait =",max(d1,d2)
        run_event.clear()
        t1.join()
        print "threads successfully closed"

if __name__ == '__main__':

    Main()