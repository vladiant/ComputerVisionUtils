# https://www.pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--image",
    type=str,
    required=True,
    help="path to input image to apply template matching",
)
ap.add_argument(
    "-t", "--template", type=str, required=True, help="path to template image"
)
ap.add_argument("-b", "--threshold", type=float, default=0.8, help="matching threshold")

args = vars(ap.parse_args())

print("Loading images")

image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# Template image spatial dimensions
(tH, tW) = template.shape[:2]

cv2.imshow("Image", image)
cv2.imshow("Template", template)

# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

print("Perform template matching")
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

(yCoords, xCoords) = np.where(result >= args["threshold"])

# Image to draw results
clone = image.copy()

print(f"{len(yCoords)} matched locations before Clustering")

# Draw matched locations before non-maxima suppression
for (x, y) in zip(xCoords, yCoords):
    # Draw bounding box
    cv2.rectangle(clone, (x, y), (x + tW, y + tH), (255, 0, 0), 1)

cv2.imshow("Before Clustering", clone)


# Clusterize
start_points = []
for (x, y) in zip(xCoords, yCoords):
    # np.append(rects, np.array([x,y, x+tW, y+tH]), axis=0)
    start_points.append((x, y))

start_points = np.array(start_points, dtype="float32")

# Define criteria = ( type, max_iter = 10 , epsilon = 0.1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans (8 labels, 10 attempts)
compactness, labels, centers = cv2.kmeans(start_points, 8, None, criteria, 10, flags)

# Draw found rectangles
for center in centers:
    startX = int(center[0])
    startY = int(center[1])
    endX = startX + tW
    endY = startY + tH
    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 1)


cv2.imshow("After Clustering", image)

cv2.waitKey(0)
