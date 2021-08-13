import sys
import cv2 as cv
import numpy as np

# Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # Ff there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding box is defined by integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain
    # in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and
        # add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x,y) coordinates for the start of
        # the bounding box and the smallest (x,y) coordinates
        # for the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        # Compute the width and the height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that
        # have overlap more than threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # Return only the bounding boxes that were picked
    # using the integer data type
    return boxes[pick].astype("int")


def f(x):
    return


strategy_types = {
    0: cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor,
    1: cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill,
    2: cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple,
    3: cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize,
    4: cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture,
}

if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} image_filename")
    exit(1)

img = cv.imread(sys.argv[1], cv.IMREAD_COLOR)

cv.namedWindow("Filtered Segment", cv.WINDOW_NORMAL)

cv.createTrackbar("Min Size", "Filtered Segment", 20, 100, f)
cv.createTrackbar("K", "Filtered Segment", 30, 100, f)
cv.createTrackbar("Sigma", "Filtered Segment", 50, 100, f)
cv.createTrackbar("Strategy Type", "Filtered Segment", 3, len(strategy_types) - 1, f)
cv.createTrackbar("Search Type", "Filtered Segment", 0, 2, f)
cv.createTrackbar("Base K", "Filtered Segment", 30, 100, f)
cv.createTrackbar("Inc K", "Filtered Segment", 30, 100, f)
cv.createTrackbar("Sigma Search", "Filtered Segment", 50, 100, f)

while True:
    # cv.setRNGSeed(1)

    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Default: double sigma=0.5, float k=300, int min_size=100
    current_min_size = cv.getTrackbarPos("Min Size", "Filtered Segment") * 10
    current_k = cv.getTrackbarPos("K", "Filtered Segment") * 10.0
    current_sigma = cv.getTrackbarPos("Sigma", "Filtered Segment") / 100.0

    segment = cv.ximgproc.segmentation.createGraphSegmentation(
        sigma=current_sigma, k=current_k, min_size=current_min_size
    )
    ss.clearGraphSegmentations()
    ss.addGraphSegmentation(segment)

    current_strategy = cv.getTrackbarPos("Strategy Type", "Filtered Segment")

    ss.clearStrategies()
    strategy = strategy_types[current_strategy]()
    ss.addStrategy(strategy)

    ss.setBaseImage(img)

    current_base_k = cv.getTrackbarPos("Base K", "Filtered Segment") * 10
    current_inc_k = cv.getTrackbarPos("Inc K", "Filtered Segment") * 10
    current_sigma_search = cv.getTrackbarPos("Sigma Search", "Filtered Segment") / 100.0

    # Options:
    # switchToSelectiveSearchFast : int base_k=150, int inc_k=150, float sigma=0.8
    # switchToSelectiveSearchQuality : int base_k=150, int inc_k=150, float sigma=0.8
    # switchToSingleStrategy : int k=200, float sigma=0.8f
    current_search = cv.getTrackbarPos("Search Type", "Filtered Segment")
    if current_search == 0:
        ss.switchToSelectiveSearchFast(
            base_k=current_base_k, inc_k=current_inc_k, sigma=current_sigma_search
        )
    elif current_search == 1:
        ss.switchToSelectiveSearchQuality(
            base_k=current_base_k, inc_k=current_inc_k, sigma=current_sigma_search
        )
    else:
        ss.switchToSingleStrategy(k=current_base_k, sigma=current_sigma_search)

    ssresults = ss.process()

    # for result in ssresults:
    #     x, y, w, h = result
    #     show_img = img.copy()
    #     cv.rectangle(show_img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
    #     cv.imshow("Segment", show_img)
    #     cv.waitKey(10)
    print(len(ssresults))

    filtered_segments = non_max_suppression_fast(ssresults, 0.9)

    show_img = img.copy()
    for result in filtered_segments:
        x, y, w, h = result
        print(result)
        cv.rectangle(show_img, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
    print()

    cv.imshow("Filtered Segment", show_img)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
