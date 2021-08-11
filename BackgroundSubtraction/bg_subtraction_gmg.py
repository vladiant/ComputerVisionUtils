import sys
import cv2 as cv


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} video_filename")
    exit(1)

cap = cv.VideoCapture(sys.argv[1])

# Background Subtractor module based on the algorithm given in
# Andrew B Godbehere, Akihiro Matsukawa, and Ken Goldberg.
# Visual tracking of human visitors under variable-lighting conditions for a responsive audio art installation.
# In American Control Conference (ACC), 2012, pages 4305–4312. IEEE, 2012.
# Default values
fgbg = cv.bgsegm.createBackgroundSubtractorGMG(
    initializationFrames=120, decisionThreshold=0.8
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
