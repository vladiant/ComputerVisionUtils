# https://www.youtube.com/watch?v=Exic9E5rNok

import sys
import os
import datetime
import time

import cv2


if len(sys.argv) <= 1:
    print(f"Format to call: {sys.argv[0]} camera_index")
    exit(os.EX_IOERR)

camera_index = int(sys.argv[1])
print(f"Opening camera index {camera_index}")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error opening camera index {camera_index}")
    exit(os.EX_IOERR)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {frame_width}")
print(f"Frame height: {frame_height}")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_READ_AFTER_DETECTION = 5

frame_size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time > SECONDS_TO_READ_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop recording!")
            else:
                timer_started = True
                detection_stopped_time = time.time()

    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    # for (x, y, width, height) in bodies:
    #     cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

    if detection:
        out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()

if out:
    out.release()

print("Done.")

cv2.destroyAllWindows()
