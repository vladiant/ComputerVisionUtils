import sys
import cv2 as cv
import numpy as np


def f(x):
    return


if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} image_filename")
    exit(1)

img = cv.imread(sys.argv[1], cv.IMREAD_COLOR)

cv.namedWindow("Segmented", cv.WINDOW_NORMAL)

cv.createTrackbar("Min Size", "Segmented", 20, 100, f)
cv.createTrackbar("K", "Segmented", 30, 100, f)
cv.createTrackbar("Sigma", "Segmented", 50, 100, f)

while True:
    # Default: double sigma=0.5, float k=300, int min_size=100
    current_min_size = cv.getTrackbarPos("Min Size", "Segmented") * 10
    current_k = cv.getTrackbarPos("K", "Segmented") * 10.0
    current_sigma = cv.getTrackbarPos("Sigma", "Segmented") / 100.0

    segmentor = cv.ximgproc.segmentation.createGraphSegmentation(
        sigma=current_sigma, k=current_k, min_size=current_min_size
    )

    # Input can be  Gray, RGB, RGB-D
    # Output type CV_32SC1
    segmented = segmentor.processImage(img)

    _, segment_count, _, _ = cv.minMaxLoc(segmented)
    print("Segments found: ", int(segment_count))

    # for segm in range(int(segment_count)):
    #     cv.imshow(f"Segment {segm}", 255 * (segm == segmented).astype(np.uint8))
    #     cv.waitKey(0)

    cv.imshow("Original", img)

    viewable = np.zeros((segmented.shape), dtype=np.uint8)

    # Normalize to be viewed
    viewable = cv.normalize(segmented, viewable, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    cv.imshow("Segmented", viewable)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
