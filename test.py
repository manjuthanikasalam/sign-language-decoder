import cv2
import numpy as np
import math
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Text-to-Speech function
def speak(label):
    engine = pyttsx3.init()
    engine.say(label)
    engine.runAndWait()

# Initialize hand detector and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/dhars/OneDrive/Desktop/converted_keras/keras_model.h5", "C:/Users/dhars/OneDrive/Desktop/converted_keras/labels.txt")

offset = 20
imgSize = 300
labels = ["Hello","I love you","Let's be friends","No","Please","Thank you","Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        speak(labels[index])

        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
