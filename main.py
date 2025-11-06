import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

# Load model and labels
model = tf.keras.models.load_model("model/hand_model.h5")

# Get labels
files = os.listdir("dataset/")
labels = [f.replace("_data.csv", "") for f in files]

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.extend([lm.x, lm.y, lm.z])

            pred = model.predict(np.array([lmList]))
            gesture = labels[np.argmax(pred)]

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            cv2.putText(img, f'{gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    cv2.imshow("Hand Movement Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
