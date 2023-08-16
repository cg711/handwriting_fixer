import cv2
import numpy as np
import pytesseract
import random
from pytesseract import Output

## USE RED SHARPIE WHEN DRAWING OUT WORDS

MAX_COUNT = 20

lines = open("wordlist.txt").read().splitlines()

currentLine = random.choice(lines)

currentCount = 0

# Facetime camera
cam = cv2.VideoCapture(1)

cv2.namedWindow("Handwriting Fixer")


# on button click read
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

    lower_red = np.array([0, 100, 20])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    if result:
        d = pytesseract.image_to_data(mask, output_type=Output.DICT)
        arr = list(map(lambda x: x.lower(), d["text"]))
        arr = list(filter(lambda x: x != "" & x != " ", arr))
        image = cv2.putText(
            image,
            currentLine,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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
                    ##arduino stuff here
        cv2.imshow("Handwriting Fixer", image)
    key = cv2.waitKey(1)
    if key == 0:
        break

cam.release()
cv2.destroyAllWindows()
