import sys

import cv2
import numpy as np

win = "poisson"


def poisson(image, seed):
    rng = np.random.default_rng(seed)
    # Determine unique values in image & calculate the next power of two
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))

    # Generating noise for each unique value in image.
    out = rng.poisson(image * vals) / float(vals)

    out = np.clip(out, 0.0, 1.0)
    return out


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

# Convert to float
img = np.float32(img) / 255.0

cv2.imshow("image", img)


def update(_):
    noised = poisson(img, 42)
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
