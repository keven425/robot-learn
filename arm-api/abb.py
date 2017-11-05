#!/usr/bin/python
# -*- coding: utf-8 -*-

import wx
import serial
import sys
from time import sleep
import random

# min, max, home
BOUNDS = {
    1: [550, 2350, 1450],
    # 2: [600, 1900, 1250],
    3: [550, 1850, 1150],
    4: [550, 1900, 1450],
    5: [550, 2450, 1500],
    6: [550, 2350, 1500],
    7: [550, 2400, 1500],
}

class Arm(object):
    def __init__(self, fps, velocity_scale=200, dry_run=False, verbose=False):
        """
        fps: the frames per second that set_multi_velocity must be called to
        keep smooth motion.
        velocity_scale: convert unitless range (-1, 1) to velocity in
        microseconds / second
        """
        self.port="/dev/ttyUSB0"

        try:
            self.dev = serial.Serial(self.port, 9600, timeout=5, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        except:
            sys.exit("Error connecting device")

        # chain = '#' + str(self.c) + 'P' + str(self.p) + 'T' + str(self.t) + '\r\n'
        # print (chain);
        # ser.write(chain)
        # # ser.write(b"#2P1900T100\r\n")
        # response =  ser.read(2)
        # print response
        # ser.close()

        self.default_speed = 100
        self.positions = self._home_position()

        self.fps = fps
        self.velocity_scale = velocity_scale

        # position scale is the velocity in (microseconds / second) / FPS to get
        # microseconds per frame
        self.position_scale = velocity_scale / self.fps

        self.dry_run = dry_run
        self.verbose = verbose

    def _home_position(self):
        return {i: BOUNDS[i][2] for i in BOUNDS}

    def _bound_position(self, axis, position):
        if position > BOUNDS[axis][1]:
            return BOUNDS[axis][1]
        if position < BOUNDS[axis][0]:
            return BOUNDS[axis][0]
        return position

    def set_position(self, axis, position, speed=None, time=None):
        """
        pos: position pulse width in microseconds
        speed: microseconds per second
        time: time in milliseconds to execute motion to `pos`
        """

        self.positions[axis] = self._bound_position(axis, position)

        if speed is None and time is None:
            speed = self.default_speed

        if self.verbose:
            print('axis=', axis)
            print('position=', position)
            print('speed=', speed)
            print('time=', time)

        if self.dry_run:
            return

        if speed:
            self.dev.write(
                '#{axis}P{position}S{speed}\r\n'.format(
                    axis=axis, position=position, speed=speed
                )
            )
        else:
            self.dev.write(
                '#{axis}P{position}T{time}\r\n'.format(
                    axis=axis, position=position, time=time
                )
            )

    def set_positions(self, positions, speeds=None, scaled=False):

        # self.dev.write("#STOP\r\n")

        if scaled:
            positions = {
                axis: self._scaled_to_absoltuion_position(axis, position)
                for axis, position in positions.items()
            }

        for axis in positions:
            positions[axis] = self._bound_position(axis, positions[axis])
            self.positions[axis] = positions[axis]

        # if speeds is None:
        #     speeds = {axis: self.default_speed for axis in positions}

        if self.verbose:
            print('positions', positions)
            print('speeds   ', speeds)

        if self.dry_run:
            return

        chain = ''.join(
                '#{axis}P{pos}T200'.format(
                    axis=axis,
                    pos=positions[axis],
                ) for axis in positions
            ) + '\r\n'

        print(chain)

        # self.dev.write(b"#2P1900T100\r\n")

        self.dev.write(chain)
        response =  self.dev.read(2)
        print response
        # self.dev.close()

    def _scaled_position(self, axis, position):
        return (position - BOUNDS[axis][0]) / (
            BOUNDS[axis][1] - BOUNDS[axis][0]
        )

    def scaled_positions(self):
        return [
            self._scaled_position(axis, position)
            for axis, position in self.positions.items()
        ]

    def _scaled_to_absoltuion_position(self, axis, position):
        if position < -1 or position > 1:
            raise ValueError((
                'position expected to be within 0 and 1.  found: {}'
            ).format(position))

        return  int(BOUNDS[axis][0] + (position + 1) / 2 * (BOUNDS[axis][1] - BOUNDS[axis][0]))

    def set_scaled_position(self, axis, position, speed=None):
        self.set_position(
            axis,
            self._scaled_to_absoltuion_position(axis, position),
            speed=speed
        )

    def set_relative_position(self, axis, position_delta, speed=None):
        self.positions[axis] += position_delta
        self.set_position(axis, self.positions[axis], speed=speed)

    def set_velocities(self, velocities):
        """
        Set velocity of all servos in arm.

        set_multi_velocity must be called once every self.fps
        """
        if set(velocities.keys()) != set(self.positions.keys()):
            raise ValueError((
                'velocities.keys must match self.positions.keys:\n'
                '  velocities.keys(): {}\n'
                '  self.position.keys(): {}\n'
            ).format(velocities.keys(), self.positions.keys()))

        if not any(v != 0 for v in velocities.values()):
            return

        self.set_positions(
            {
                axis: self.positions[axis] + (velocity * self.position_scale)
                for axis, velocity in velocities.items()
            },
            {
                axis: max(abs(velocity * self.velocity_scale), 100)
                for axis, velocity in velocities.items()
            },
        )

    def set_velocity(self, axis, velocity):
        velocity *= 10
        self.set_position(
            axis,
            self.positions[axis] + velocity * self.position_scale,
            speed=max(abs(velocity) * self.velocity_scale, 5)
        )

    def go_home(self):
        self.set_positions(self._home_position())

class ServoControl(wx.Frame):

  def __init__(self, *args, **kw):
    super(ServoControl, self).__init__(*args, **kw)
    self.root=sys.path[0]
    self.port="/dev/ttyUSB0"
    self.channels=8
    self.InitUI()
    # ser = serial.Serial(self.Port)
    # ser.baudrate = 9600
    # ser.timeout = 1
    # ser.write(b"#1P1000T100")
    # print(ser)
    # ser.close()

  def InitUI(self):

    pnl = wx.Panel(self)
    self.label = wx.StaticText(pnl, label='Port :', pos=(20,85))
    self.portwg = wx.TextCtrl(pnl, -1, self.port, size=(175, -1), pos=(100, 80))
    self.portwg.Bind(wx.EVT_TEXT, self.OnPortChange)

    self.channel = []
    for i in range(1,self.channels):
      self.channel.append(channel(self,i, 2000, 100, pnl))

    self.SetSize((290, 100 + 50 * self.channels))
    self.SetTitle('ServoToolBox')
    self.Centre()
    self.Show(True)

  def OnPortChange(self, e):
    self.port = self.portwg.GetValue()


# end of class SerialConfigDialog

class channel():
  ''' Classe canal.'''
  def __init__(self, parent, channel, position, time, pnl):
    self.parent=parent
    self.c=channel
    self.p=position
    self.t=time
    ''' Entropie '''
    self.dp=10
    ''' Balance '''
    self.dps=500
    self.slider = wx.Slider(pnl, value=100, minValue=0, maxValue=200, pos=(20, 80 + ( 40 * self.c )), size=(250, 15), style=wx.SL_HORIZONTAL)
    self.label = wx.StaticText(pnl, label='Canal ' + str(self.c) + ' - Position: ' + str(self.p)+'°', pos=(20, 100 + ( 40 * self.c )))
    self.slider.Bind(wx.EVT_SCROLL, self.OnSliderScroll)

  def OnSliderScroll(self, e):
    self.p = self.slider.GetValue() * self.dp + self.dps
    self.label.SetLabel('Canal ' + str(self.c) + ' - Position: ' + str(self.p)+'°')
    # ser = serial.Serial(self.parent.Port, 9600)
    try:
        ser = serial.Serial(self.parent.port, 9600, timeout=5, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
    except:
        sys.exit("Error connecting device")

    chain = '#' + str(self.c) + 'P' + str(self.p) + 'T' + str(self.t) + '\r\n'
    print (chain);
    ser.write(chain)
    # ser.write(b"#2P1900T100\r\n")
    response =  ser.read(2)
    print response
    ser.close()

class MyApp(wx.App):

    ex = wx.App()
    ServoControl(None)


if __name__ == '__main__':

    app = MyApp(0)
    app.MainLoop()

    arm = Arm(5)
    arm.go_home()

    # sleep(3)

    # positions = {
    #     1: -0.5,
    #     # 2: 0,
    #     3: 0,
    #     4: -0.3,
    #     5: 0.3,
    #     6: 1,
    #     7: -0.8,
    # }
    # arm.set_positions(positions, scaled=True)
