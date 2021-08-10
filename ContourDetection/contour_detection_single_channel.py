# https://learnopencv.com/contour-detection-using-opencv-python-c/

import cv2

# Read the image
image = cv2.imread("OpenCV_Logo.png")

# RGB splitting
blue, green, red = cv2.split(image)

# Apply the binary thresholding
ret_blue, thresh_blue = cv2.threshold(blue, 150, 255, cv2.THRESH_BINARY)
ret_green, thresh_green = cv2.threshold(green, 150, 255, cv2.THRESH_BINARY)
ret_red, thresh_red = cv2.threshold(red, 150, 255, cv2.THRESH_BINARY)

# Detect the contours
contours_blue, hierarchy_blue = cv2.findContours(
    image=thresh_blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
)
contours_green, hierarchy_green = cv2.findContours(
    image=thresh_green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
)
contours_red, hierarchy_red = cv2.findContours(
    image=thresh_red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
)

# Draw contours on the original image
image_copy_blue = image.copy()
image_copy_red = image.copy()
image_copy_green = image.copy()
cv2.drawContours(
    image=image_copy_blue,
    contours=contours_blue,
    contourIdx=-1,
    color=(0, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA,
)
cv2.drawContours(
    image=image_copy_red,
    contours=contours_red,
    contourIdx=-1,
    color=(0, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA,
)
cv2.drawContours(
    image=image_copy_green,
    contours=contours_green,
    contourIdx=-1,
    color=(0, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA,
)

# Show the contours
cv2.imshow("None approximation Blue", image_copy_blue)
cv2.imshow("None approximation Red", image_copy_red)
cv2.imshow("None approximation Green", image_copy_green)

cv2.waitKey(0)
