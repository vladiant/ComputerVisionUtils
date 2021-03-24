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

    # upload resized frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # convert to gray
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # upload pre-processed frame to GPU
    gpu_previous = cv2.cuda_GpuMat()
    gpu_previous.upload(previous_frame)

    # create gpu_hsv output for optical flow
    gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
    gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

    gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
    gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
    gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)

    # set saturation to 1
    gpu_s.upload(np.ones_like(previous_frame, np.float32))


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

    # upload frame to GPU
    gpu_frame.upload(frame)

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
    gpu_frame = cv2.cuda.resize(gpu_frame, (960, 540))

    # convert to gray
    gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

    # end pre-process timer
    end_pre_time = time.time()

    # add elapsed iteration time
    timers["pre-process"].append(end_pre_time - start_pre_time)

    # start optical flow timer
    start_of = time.time()

    # create optical flow instance
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
        5,
        0.5,
        False,
        15,
        3,
        5,
        1.2,
        0,
    )
    # calculate optical flow
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
        gpu_flow,
        gpu_previous,
        gpu_current,
        None,
    )

    # end of timer
    end_of = time.time()

    # add elapsed iteration time
    timers["optical flow"].append(end_of - start_of)

    # start post-process timer
    start_post_time = time.time()

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x,
        gpu_flow_y,
        angleInDegrees=True,
    )

    # set value to normalized magnitude from 0 to 1
    gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)

    # get angle of optical flow
    angle = gpu_angle.download()
    angle *= (1 / 360.0) * (180 / 255.0)

    # set hue according to the angle of optical flow
    gpu_h.upload(angle)

    # merge h,s,v channels
    cv2.cuda.merge([gpu_h, gpu_s, gpu_v], gpu_hsv)

    # multiply each pixel value to 255
    gpu_hsv.convertTo(cv2.CV_8U, 255.0, gpu_hsv_8u, 0.0)

    # convert hsv to bgr
    gpu_bgr = cv2.cuda.cvtColor(gpu_hsv_8u, cv2.COLOR_HSV2BGR)

    # send original frame from GPU back to CPU
    frame = gpu_frame.download()

    # send result from GPU back to CPU
    bgr = gpu_bgr.download()

    # update previous_frame value
    gpu_previous = gpu_current

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
