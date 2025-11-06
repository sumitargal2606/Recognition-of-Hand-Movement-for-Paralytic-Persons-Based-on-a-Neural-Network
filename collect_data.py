import cv2
import mediapipe as mp
import pandas as pd
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

data = []
labels = []

gesture = input("Enter gesture name (e.g. left, right, up, down, open, fist): ")

cap = cv2.VideoCapture(0)
count = 0

if not os.path.exists("dataset"):
    os.makedirs("dataset")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for lm in handLms.landmark:
                lmList.extend([lm.x, lm.y, lm.z])
            data.append(lmList)
            labels.append(gesture)
            count += 1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, f'Samples: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collecting Data", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(f'dataset/{gesture}_data.csv', index=False)
print(f"✅ Data for '{gesture}' saved successfully!")
