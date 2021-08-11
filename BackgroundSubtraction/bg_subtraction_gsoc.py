import sys
import cv2 as cv


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

cap = cv.VideoCapture(sys.argv[1])

# Implementation of the different yet better algorithm which is called GSOC,
# as it was implemented during GSOC and was not originated from any paper
# Default values https://docs.opencv.org/4.5.1/d2/d55/group__bgsegm.html#ga7ba3e826c343adc15782ab9139f82445
fgbg = cv.bgsegm.createBackgroundSubtractorGSOC()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)

    cv.namedWindow("original", cv.WINDOW_NORMAL)
    cv.imshow("original", frame)

    cv.namedWindow("mask", cv.WINDOW_NORMAL)
    cv.imshow("mask", fgmask)

    k = cv.waitKey(1)
    if k == 27:
        break
