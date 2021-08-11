import sys
import cv2 as cv


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

cap = cv.VideoCapture(sys.argv[1])

# Background subtraction based on counting
# Default values
fgbg = cv.bgsegm.createBackgroundSubtractorCNT(
    minPixelStability=15, useHistory=True, maxPixelStability=15 * 60, isParallel=True
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
