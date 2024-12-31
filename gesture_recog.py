import cv2
import numpy as np
import math
import pickle as pkl
import os
import mediapipe as mp

w, h = 640, 480

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

keypoints = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

tol = 15

file_input = input("Do you want Alphabets or Numbers, provide A or N as input : ").upper()
if file_input == 'A':
    file_name = 'gesture.pkl'
    detection_color = (128, 0, 128)  # Purple for alphabets
else:
    file_name = 'gestures.pkl'
    detection_color = None  # Colors will be set dynamically for numbers

if os.path.exists(file_name):
    with open(file_name, 'rb') as f:
        gestnames = pkl.load(f)
        knowngesture = pkl.load(f)
else:
    print(f"File '{file_name}' does not exist.")
    exit()


class Handmarks:
    def __init__(self, hc=2, tol1=0.5, tol2=0.5):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=hc, min_detection_confidence=tol1, min_tracking_confidence=tol2)
        self.h_draw = mp.solutions.drawing_utils

    def h_marks(self, val):
        valrgb = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)
        results = self.hands.process(valrgb)
        hand_marks = []
        if results.multi_hand_landmarks:
            for marks in results.multi_hand_landmarks:
                hand_mark = [(int(mark.x * w), int(mark.y * h)) for mark in marks.landmark]
                hand_marks.append(hand_mark)
        return hand_marks


def finddistances(handdata):
    distmatrix = np.zeros((len(handdata), len(handdata)), dtype='float')
    palm_size = ((handdata[0][0] - handdata[9][0]) ** 2 + (handdata[0][1] - handdata[9][1]) ** 2)
    for i in range(len(handdata)):
        for j in range(len(handdata)):
            distmatrix[i][j] = math.sqrt(
                ((handdata[i][0] - handdata[j][0]) ** 2 + (handdata[i][1] - handdata[j][1]) ** 2) / palm_size)
    return distmatrix


def find_error(knownmatrix, unknownmatrix, keypoints):
    error = 0
    for i in keypoints:
        for j in keypoints:
            error += abs(knownmatrix[i][j] - unknownmatrix[i][j])
    return error


def findgesture(unknowngest, knowngest, keypoints, tol, gestnames):
    errorarray = []
    for i in range(len(gestnames)):
        error = find_error(knowngest[i], unknowngest, keypoints)
        errorarray.append(error)
    errormin = min(errorarray)
    minindex = errorarray.index(errormin)
    if errormin < tol:
        gesture = gestnames[minindex]
    else:
        gesture = 'UNKNOWN'
    return gesture


# Create a dictionary to map numbers to specific colors
gesture_colors = {
    '1': (0, 0, 255),    # Red
    '2': (0, 255, 0),    # Green
    '3': (255, 0, 0),    # Blue
    '4': (0, 255, 255),  # Yellow
    '5': (255, 255, 0),  # Cyan
    '6': (255, 0, 255),  # Purple
    '7': (255, 165, 0),  # Orange
    '8': (128, 0, 128),  # Purple (dark)
    '9': (128, 128, 0),  # Olive
    '0': (0, 0, 0)       # Black for Zero
}

handmarks = Handmarks()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    marks = handmarks.h_marks(frame)

    if marks:
        # Get the unknown hand's distance matrix
        unknownmatrix = finddistances(marks[0])
        
        # Find the detected gesture
        find_gesture = findgesture(unknownmatrix, knowngesture, keypoints, tol, gestnames)
        
        # Choose the color based on input type
        if detection_color:  # For alphabets, use purple
            gesture_color = detection_color
        else:  # For numbers, pick from the dictionary
            gesture_color = gesture_colors.get(find_gesture, (255, 255, 255))  # Default to white if unknown
        
        # Display "Gesture Detected: [gesture]" on the frame with the selected color
        gesture_message = f"Gesture Detected: {find_gesture}"
        cv2.putText(frame, gesture_message, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, gesture_color, 2)

    # Show the frame with gesture detection message
    cv2.imshow('Gesture Recognition', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
