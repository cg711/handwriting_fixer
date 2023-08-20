import cv2 # opencv-python
import numpy as np
import pytesseract as pt # tesseract
from pytesseract import Output
import random
import time

# arduino 
import serial # pyserial

MAX_COUNT = 50

# word bank init
lines = open("wordlist.txt").read().splitlines()
currentLine = random.choice(lines)
currentCount = 0

# HSV bounds
lower_green = np.array([40, 100, 100])
upper_green = np.array([100, 255, 255])

# Facetime camera
cam = cv2.VideoCapture(0)
cv2.namedWindow("Handwriting Fixer")

# aruino config
uno = serial.Serial(port="/dev/cu.usbmodem23401", baudrate=9600, timeout=0.1)

def onRecognize(d):
    global currentLine
    print(d)
    if currentLine in d:
        currentLine = random.choice(lines)
        return True
    else:
        return False


while True:
    result, image = cam.read()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    if result:
        d = pt.image_to_data(mask, output_type=Output.DICT)
        arr = list(map(lambda x: x.lower(), d["text"]))
        arr = list(filter(lambda x: (x != '') & (x != ' ') & (x.isalnum()), arr)) #(not x.isspace())
        image = cv2.putText(
            mask,
            currentLine,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        res = False
        if len(arr) != 0:
            if onRecognize(arr):
                print("Correct")
                currentCount = 0
            else:
                print("Incorrect")
                currentCount += 1
                if currentCount >= MAX_COUNT:
                    currentCount = 0
                    print("spray time")
                    uno.write(bytes(0x1))
        # conc = np.vstack((mask, image))
        cv2.imshow("Handwriting Fixer", image)
    key = cv2.waitKey(1)
    if key == 0:
        break

cam.release()
cv2.destroyAllWindows()
