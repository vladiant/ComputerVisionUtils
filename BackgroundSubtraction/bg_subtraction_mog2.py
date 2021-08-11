import sys
import cv2 as cv


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

cap = cv.VideoCapture(sys.argv[1])

# Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
# Default values
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

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
