import sys
import cv2 as cv


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

cap = cv.VideoCapture(sys.argv[1])

# Background Subtraction using Local SVD Binary Pattern.
# https://docs.opencv.org/4.5.1/d2/d55/group__bgsegm.html#gaec3834f6a43f5bfda4d6bdd23c44d394
fgbg = cv.bgsegm.createBackgroundSubtractorLSBP()

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
