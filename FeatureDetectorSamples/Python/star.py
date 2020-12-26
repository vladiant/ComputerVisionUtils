# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md
# Source: https://docs.opencv.org/3.4.4/dc/d7d/tutorial_py_brief.html
# pip3 install opencv-contrib-python

import cv2 as cv

img = cv.imread('Lenna.png')

# Initiate STAR detector
star = cv.xfeatures2d.StarDetector_create()

# find the keypoints with STAR
kp = star.detect(img, None)

img_kp = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv.imshow("STAR keypoints", img_kp)
cv.waitKey(0)
cv.destroyAllWindows()
