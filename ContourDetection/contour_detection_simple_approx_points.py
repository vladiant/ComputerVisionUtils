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
contours, hierarchy = cv2.findContours(
    image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
)

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

# Draw contour points
image_copy_2 = image.copy()
for i, contour in enumerate(contours):  # Loop over single contour
    for j, contour_point in enumerate(contour):  # Loop over contour points
        cv2.circle(
            image_copy_2,
            ((contour_point[0][0], contour_point[0][1])),
            2,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

# Show the contours
cv2.imshow("Simple approximation", image_copy)
cv2.imshow("Simple approximation points", image_copy_2)

cv2.waitKey(0)
