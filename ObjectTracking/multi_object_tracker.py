# Based on https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import sys
import os
import numpy as np
import cv2 as cv

"""
https://github.com/methylDragon/opencv-python-reference/blob/master/03%20OpenCV%20Analysis%20and%20Object%20Tracking.md
My personal suggestion is to:

Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
Use MOSSE when you need pure speed
"""

OPENCV_OBJECT_TRACKERS = {
    # Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. (minimum OpenCV 3.4.2)
    "csrt": cv.TrackerCSRT_create,
    # Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. (minimum OpenCV 3.1.0)
    "kcf": cv.TrackerKCF_create,
    # BOOSTING Tracker: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost),
    # but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well.
    # Interesting only for legacy reasons and comparing other algorithms. (minimum OpenCV 3.0.0)
    "boosting": cv.TrackerBoosting_create,
    # Better accuracy than BOOSTING tracker but does a poor job of reporting failure. (minimum OpenCV 3.0.0)
    "mil": cv.TrackerMIL_create,
    # I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual algorithm itself,
    # but the TLD tracker was incredibly prone to false-positives. I do not recommend using this OpenCV object tracker. (minimum OpenCV 3.0.0)
    "tld": cv.TrackerTLD_create,
    # Does a nice job reporting failures; however, if there is too large of a jump in motion,
    # such as fast moving objects, or objects that change quickly in their appearance, the model will fail. (minimum OpenCV 3.0.0)
    "medianflow": cv.TrackerMedianFlow_create,
    # Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed. (minimum OpenCV 3.4.1)
    "mosse": cv.TrackerMOSSE_create,
}

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

# Create multi objects tracker
trackers = cv.MultiTracker_create()

# Set tracker
tracker1 = OPENCV_OBJECT_TRACKERS["csrt"]()
tracker2 = OPENCV_OBJECT_TRACKERS["csrt"]()

# Setup initial location of window
# Valid for https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
trackers.add(tracker1, frame, (262, 78, 50, 45))
trackers.add(tracker2, frame, (300, 200, 100, 50))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, new_bounding_boxes = trackers.update(frame)
    if success:
        x1 = int(new_bounding_boxes[0][0])
        y1 = int(new_bounding_boxes[0][1])
        w1 = int(new_bounding_boxes[0][2])
        h1 = int(new_bounding_boxes[0][3])

        x2 = int(new_bounding_boxes[1][0])
        y2 = int(new_bounding_boxes[1][1])
        w2 = int(new_bounding_boxes[1][2])
        h2 = int(new_bounding_boxes[1][3])

        img2 = frame.copy()

        cv.rectangle(img2, (x1, y1), (x1 + w1, y1 + h1), 255, 2)
        cv.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), 127, 2)

        cv.imshow("Frame", img2)
    else:
        cv.imshow("Frame", frame)

    if cv.waitKey(10) & 0xFF == 27:
        break

cap.release()

print("Done.")

cv.destroyAllWindows()
