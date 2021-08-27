# https://www.pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/

import numpy as np
import argparse
import cv2

# Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

    # idxs = np.delete(
    #     idxs, np.concatenate(([last], np.logical_or(overlap > overlapThresh, overlap < 0.7)))
    # )


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

print(f"{len(yCoords)} matched locations before NMS")

# Draw matched locations before non-maxima suppression
for (x, y) in zip(xCoords, yCoords):
    # Draw bounding box
    cv2.rectangle(clone, (x, y), (x + tW, y + tH), (255, 0, 0), 1)

cv2.imshow("Before NMS", clone)

# Initial list of rectangles
rects = []

# Loop over starting coordinates
for (x, y) in zip(xCoords, yCoords):
    # np.append(rects, np.array([x,y, x+tW, y+tH]), axis=0)
    rects.append((x, y, x + tW, y + tH))

rects = np.array(rects)

# Apply non-maxima suppression
pick = non_max_suppression_fast(rects, 0.2)
print(f"{len(pick)} matched locations after NMS")

# Loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 1)

cv2.imshow("After NMS", image)

cv2.waitKey(0)
