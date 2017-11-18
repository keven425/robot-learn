/*
 * ------------------------------
 *   MultipleSerialServoControl
 * ------------------------------
 *
 * Uses the Arduino Serial library
 *  (http://arduino.cc/en/Reference/Serial)
 * and the Arduino Servo library
 *  (http://arduino.cc/en/Reference/Servo)
 * to control multiple servos from a PC using a USB cable.
 *
 * Dependencies:
 *   Arduino 0017 or higher
 *     (http://www.arduino.cc/en/Main/Software)
 *   Python servo.py module
 *     (http://principialabs.com/arduino-python-4-axis-servo-control/)
 *
 * Created:  23 December 2009
 * Author:   Brian D. Wendt
 *   (http://principialabs.com/)
 * Version:  1.1
 * License:  GPLv3
 *   (http://www.fsf.org/licensing/)
 *
 */

// Import the Arduino Servo library
#include <Servo.h>

// Create a Servo object for each servo
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;

// TO ADD SERVOS:
//   Servo servo5;
//   etc...

// Common servo setup values
int minPulse = 600;   // minimum servo position, us (microseconds)
int maxPulse = 2400;  // maximum servo position, us

// User input for servo and position
int userInput[6];    // raw input from serial buffer, 3 bytes
int startbyte;       // start byte, begin reading input
int servo;           // which servo to pulse?
int pos;             // servo angle 0-180
int i;               // iterator

// LED on Pin 13 for digital on/off demo
int ledPin = 13;
int pinState = LOW;

// Calibration values
int feedbackPin1 = A0;
int feedbackPin2 = A1;
int feedbackPin3 = A2;
int feedbackPin4 = A3;
int feedbackPin5 = A4;
int feedbackPin6 = A5;

// Calibration values
int minDegrees;
int maxDegrees;
int minFeedback;
int maxFeedback;
int tolerance = 2; // max feedback measurement error

int val1 = 0;
int val2 = 0;
int val3 = 0;
int val4 = 0;
int val5 = 0;
int val6 = 0;

void setup()
{
    // Attach each Servo object to a digital pin
    servo1.attach(3, minPulse, maxPulse);
    servo2.attach(5, minPulse, maxPulse);
    servo3.attach(6, minPulse, maxPulse);
    servo4.attach(9, minPulse, maxPulse);
    servo5.attach(10, minPulse, maxPulse);
    servo6.attach(11, minPulse, maxPulse);
    // TO ADD SERVOS:
    //   servo5.attach(YOUR_PIN, minPulse, maxPulse);
    //   etc...

//    calibrate(servo1, feedbackPin1, 20, 160);  // calibrate for the 20-160 degree range

    // LED on Pin 13 for digital on/off demo
    pinMode(ledPin, OUTPUT);

    // Open the serial connection, 9600 baud
    Serial.begin(9600);

    while (!Serial.available());

    startbyte = -1;
    while (Serial.available() > 0 && startbyte != 254) {
      startbyte = Serial.read();
//      Serial.print("startbyte: ");
//      Serial.println(startbyte);
    }
    Serial.print("got 254. serial ready.");
}

/*
    This function establishes the feedback values for 2 positions of the servo.
    With this information, we can interpolate feedback values for intermediate positions
*/
void calibrate(Servo servo, int analogPin, int minPos, int maxPos)
{
    // Move to the minimum position and record the feedback value
    servo.write(minPos);
    minDegrees = minPos;
    delay(2000); // make sure it has time to get there and settle
    minFeedback = analogRead(analogPin);

    // Move to the maximum position and record the feedback value
    servo.write(maxPos);
    maxDegrees = maxPos;
    delay(2000); // make sure it has time to get there and settle
    maxFeedback = analogRead(analogPin);
}

void feedback ()
{
    val1 = analogRead(feedbackPin1);            // reads the value of the potentiometer (value between 0 and 1023)
    val2 = analogRead(feedbackPin2);
    val3 = analogRead(feedbackPin3);
    val4 = analogRead(feedbackPin4);
    val5 = analogRead(feedbackPin5);
    val6 = analogRead(feedbackPin6);

    // val = map(val, 139, 500, 0, 179);     // scale it to use it with the servo (value between 0 and 180)

    char str[1000];
    sprintf(str, "%d,%d,%d,%d,%d,%d\n", val1, val2, val3, val4, val5, val6);
    Serial.println(str);
}

void loop()
{
//    while (!Serial.available());
//    while (Serial.available() > 0 && startbyte != 255) {
//      startbyte = Serial.read();
//    }
    if (Serial.available() > 6) {
        // Read the first byte
        startbyte = Serial.read();
        if (startbyte != 255) {
          exit(1); // not good
          return;
        }
        // ... then get the next two bytes
        for (i=0;i<6;i++) {
            userInput[i] = Serial.read();
        }
        servo1.write(userInput[0]);
        servo2.write(userInput[1]);
        servo3.write(userInput[2]);
        servo4.write(userInput[3]);
        servo5.write(userInput[4]);
        servo6.write(userInput[5]);
        while (true) {
          Serial.println("200,200,200,200,200,200");
        }
    }

//            while (true) {
//                feedback();
//            }

//            for (i = 0; i < 6; i++) {
//              Serial.print("servo");
//              Serial.print(i);
//              Serial.print(": ");
//              Serial.print(userInput[i]);
//              Serial.print("\n\n");
//            }
//            userInput[0];
          // Packet error checking and recovery
//            if (pos == 255 || servo == 255 || !pos || !servo) { return; }
//    }
//    Serial.println("end of loop");
//    feedback();
}
