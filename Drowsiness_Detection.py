import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import imutils
import time
import dlib
import os
from pygame import mixer

mixer.init()  # Initializing the mixer module from Pygame
mixer.music.load("music.wav")


def alarm(msg):
    global alarm_status
    global alarm_status2
    global alarm_status3
    global saying

    while alarm_status:
        mixer.music.play()
        print('call')
        s = 'espeak "' + msg + '"'
        os.system(s)

    while alarm_status2:
        mixer.music.play()
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

    while alarm_status3:
        mixer.music.play()
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


def head_angle(shape):
    left_eyebrow_chin_dist = np.linalg.norm(shape[19] - shape[8])
    eyebrow_chin_dist = np.linalg.norm(shape[21] - shape[8])

    angle_1 = np.arccos(
        np.dot((shape[21] - shape[8]), (shape[19] - shape[8])) / (
                2 * eyebrow_chin_dist * left_eyebrow_chin_dist))
    angle_1 = np.degrees(angle_1)

    return angle_1


def preprocess_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    return equalized


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 25
HEAD_ANGLE_THRESH = 60.7  # Adjust this value as needed

alarm_status = False
alarm_status2 = False
alarm_status3 = False
saying = False
COUNTER = 0
YAWN_COUNTER = 0
YAWN_FRAME = 20
HEAD_COUNTER = 0
HEAD_FRAME = 20

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        # Create a new variable to hold landmarks for each face
        face_landmarks = shape

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = face_landmarks[48:60]  # Use the new variable here
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        angle = head_angle(face_landmarks)  # Use the new variable here

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('Wake up, sir!',))
                    t.daemon = True
                    t.start()
                    print('DROWSINESS ALERT!')

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            YAWN_COUNTER += 1

            if YAWN_COUNTER >= YAWN_FRAME:
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('Take some fresh air, sir!',))
                    t.daemon = True
                    t.start()
                    print('YAWN ALERT!')

                cv2.putText(frame, "YAWN ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            YAWN_COUNTER = 0
            alarm_status2 = False

        if angle > HEAD_ANGLE_THRESH:
            HEAD_COUNTER += 1

            if HEAD_COUNTER >= HEAD_FRAME:
                if alarm_status3 == False and saying == False:
                    alarm_status3 = True
                    t = Thread(target=alarm, args=('Pay attention to the road, sir!',))
                    t.daemon = True
                    t.start()
                    print('HEAD ANGLE ALERT!')

                cv2.putText(frame, "HEAD ANGLE ALERT!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            HEAD_COUNTER = 0
            alarm_status3 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "A: {:.2f}".format(angle), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
