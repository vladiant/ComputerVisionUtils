# https://learnopencv.com/contour-detection-using-opencv-python-c/

import cv2

# Read the image
image = cv2.imread("OpenCV_Logo.png")

# Convert image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# Show the binary image
cv2.imshow("Binary Image", thresh)

# Detect the contours
# Method does not create any parent child relationship between the extracted contours.
contours, hierarchy = cv2.findContours(
    image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
)

# [Next, Previous, First_Child, Parent]
print(hierarchy)

# Draw contours on the original image
image_copy = image.copy()
cv2.drawContours(
    image=image_copy,
    contours=contours,
    contourIdx=-1,
    color=(0, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA,
)

# Show the contours
cv2.imshow("RETR_LIST", image_copy)

cv2.waitKey(0)
