# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as npjjjj
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
goodScanPersentechStop = 0
goodScanPersentechF = 0
# goodScanPersentechVolumedown = 0
# goodScanPersentechVolumeUp = 0

sensibility = 10


def setControl(classData):
    global goodScanPersentechStop,  goodScanPersentechF

    if classData == 'stop' or classData == 'live long':
        goodScanPersentechStop = goodScanPersentechStop + 1
        goodScanPersentechF = 0
        if goodScanPersentechStop > sensibility:
            pyautogui.press('space')
            goodScanPersentechStop = 0

    if classData == 'rock':
        goodScanPersentechF = goodScanPersentechF + 1
        goodScanPersentechStop = 0
        if goodScanPersentechF > sensibility:
            pyautogui.press('j')
            goodScanPersentechF = 0

    if classData == 'thumbs up':
        pyautogui.press('volumeup')
        goodScanPersentechStop = 0
        goodScanPersentechF = 0

    if classData == 'thumbs down':
        pyautogui.press('volumedown')
        goodScanPersentechStop = 0
        goodScanPersentechF = 0


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            setControl(className)
    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(goodScanPersentechStop), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
