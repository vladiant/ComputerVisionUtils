# Based on https://docs.opencv.org/3.4.4/db/df8/tutorial_py_meanshift.html

import sys
import os
import numpy as np
import cv2 as cv

if len(sys.argv) <= 1:
    print(f"Format to call: {sys.argv[0]} video_stream_file/camera_index")
    exit(os.EX_IOERR)

video_stream = sys.argv[1]

print(f"Opening video stream {video_stream}")

cap = cv.VideoCapture(video_stream)

if not cap.isOpened():
    print(f"Error opening video stream: {video_stream}")

    camera_index = int(sys.argv[1])
    print(f"Opening camera index {camera_index}")
    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error opening camera index {camera_index}")
        exit(os.EX_IOERR)

cv.namedWindow("Frame", cv.WINDOW_NORMAL)

# Take first frame from the video
ret, frame = cap.read()
if not ret:
    print("Error reading first frame")
    exit(os.EX_IOERR)

# Setup initial location of window
# Valid for https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

# Set up the ROI for tracking
# Valid for https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
roi = frame[y : y + h, x : x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move at least 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply meanshift to get new location
    ret, track_window = cv.meanShift(dst, track_window, term_crit)
    # print(ret) # iterations to converge

    # Draw on image
    x, y, w, h = track_window
    img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    cv.imshow("Frame", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cap.release()

print("Done.")

cv.destroyAllWindows()
