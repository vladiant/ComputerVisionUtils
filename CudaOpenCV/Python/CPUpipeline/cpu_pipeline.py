# https://learnopencv.com/getting-started-opencv-cuda-module/

import time
import sys

import cv2
import numpy as np


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

# init video capture with video
cap = cv2.VideoCapture(sys.argv[1])

# get default video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# get total number of video frames
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("Number of frames:", num_frames)

# read the first frame
ret, previous_frame = cap.read()

# proceed if frame reading was successful
if ret:
    # resize frame
    frame = cv2.resize(previous_frame, (960, 540))

    # convert to gray
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create hsv output for optical flow
    hsv = np.zeros_like(frame, np.float32)

    # set saturation to 1
    hsv[..., 1] = 1.0

timers = dict()
timers["reading"] = list()
timers["pre-process"] = list()
timers["optical flow"] = list()
timers["post-process"] = list()
timers["full pipeline"] = list()

while True:
    # start full pipeline timer
    start_full_time = time.time()

    # start reading timer
    start_read_time = time.time()

    # capture frame-by-frame
    ret, frame = cap.read()

    # end reading timer
    end_read_time = time.time()

    # add elapsed iteration time
    timers["reading"].append(end_read_time - start_read_time)

    # if frame reading was not successful, break
    if not ret:
        break

    # start pre-process timer
    start_pre_time = time.time()
    # resize frame
    frame = cv2.resize(frame, (960, 540))

    # convert to gray
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # end pre-process timer
    end_pre_time = time.time()

    # add elapsed iteration time
    timers["pre-process"].append(end_pre_time - start_pre_time)

    # start optical flow timer
    start_of = time.time()

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame,
        current_frame,
        None,
        0.5,
        5,
        15,
        3,
        5,
        1.2,
        0,
    )
    # end of timer
    end_of = time.time()

    # add elapsed iteration time
    timers["optical flow"].append(end_of - start_of)

    # start post-process timer
    start_post_time = time.time()

    # convert from cartesian to polar coordinates to get magnitude and angle
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True,
    )

    # set hue according to the angle of optical flow
    hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

    # set value according to the normalized magnitude of optical flow
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
        -1,
    )

    # multiply each pixel value to 255
    hsv_8u = np.uint8(hsv * 255.0)

    # convert hsv to bgr
    bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    # update previous_frame value
    previous_frame = current_frame

    # end post-process timer
    end_post_time = time.time()

    # add elapsed iteration time
    timers["post-process"].append(end_post_time - start_post_time)

    # end full pipeline timer
    end_full_time = time.time()

    # add elapsed iteration time
    timers["full pipeline"].append(end_full_time - start_full_time)

    # visualization
    cv2.imshow("original", frame)
    cv2.imshow("result", bgr)
    k = cv2.waitKey(1)
    if k == 27:
        break

# elapsed time at each stage
print("Elapsed time")
for stage, seconds in timers.items():
    print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

# calculate frames per second
print("Default video FPS : {:0.3f}".format(fps))

of_fps = (num_frames - 1) / sum(timers["optical flow"])
print("Optical flow FPS : {:0.3f}".format(of_fps))

full_fps = (num_frames - 1) / sum(timers["full pipeline"])
print("Full pipeline FPS : {:0.3f}".format(full_fps))
