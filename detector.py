import cv2
import numpy as np
import pytesseract
import random
from pytesseract import Output

lines = open("wordlist.txt").read().splitlines()

currentLine = random.choice(lines)

# Facetime camera
cam = cv2.VideoCapture(1)

cv2.namedWindow("Handwriting Fixer")


# on button click read
def onRecognize(d, currentLine):
    if d[0] == currentLine:
        print("Good")
    else:
        print("Bad")
    currentLine = random.choice(lines)


while True:
    result, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if result:
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        arr = list(filter(lambda x: x != "", d["text"]))
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
        cv2.imshow("Handwriting Fixer", image)
        print(arr)
        if len(arr) != 0:
            onRecognize(arr, currentLine)

    key = cv2.waitKey(1)
    if key == 0:
        break

cam.release()
cv2.destroyAllWindows()

# for i in range(len(d["text"])):
#     print(d["text"][i])
