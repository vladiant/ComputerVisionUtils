# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md

import cv2 as cv

img = cv.imread('Lenna.png')

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create(threshold=50000)

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

# Keypoints definition
# cv.KeyPoint(x, y, _size[, _angle[, _response[, _octave[, _class_id]]]])
# Or have the following members
# angle
# class_id
# convert()
# octave
# overlap()
# pt
# response
# size
# points2f = cv.KeyPoint_convert(keypoints[, keypointIndexes])

print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
# Flags:
# cv.DRAW_MATCHES_FLAGS_DEFAULT,
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
# cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
# cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv.imshow('FAST with suppression', img2)
cv.imshow('FAST no suppression', img3)

cv.waitKey(0)
cv.destroyAllWindows()
