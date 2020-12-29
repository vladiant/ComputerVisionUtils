# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md
# Source: https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html

import cv2 as cv

img1 = cv.imread('box.png')  # queryImage
img2 = cv.imread('box_in_scene.png')  # trainImage

# Initiate ORB detector
orb = cv.ORB_create(WTA_K=3)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
# Recommended for ORB WTA_K=3 and 4 
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Flags:
# cv.DRAW_MATCHES_FLAGS_DEFAULT
# cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv.namedWindow('ORB WTA3 BF Matcher', cv.WINDOW_NORMAL)
cv.imshow('ORB WTA3 BF Matcher', img3)

cv.waitKey(0)
