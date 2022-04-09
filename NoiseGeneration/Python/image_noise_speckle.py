import sys

import cv2
import numpy as np

win = "speckle"


def speckle(image, mean, var, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, var ** 0.5, image.shape)
    out = image + image * noise

    out = np.clip(out, 0.0, 1.0)
    return out


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

# Convert to float
img = np.float32(img) / 255.0

cv2.imshow("image", img)


def update(_):
    mean = cv2.getTrackbarPos("mean", win) / 100.0 - 0.5
    var = cv2.getTrackbarPos("var", win) / 2000.0
    noised = speckle(img, mean, var, 42)
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


cv2.createTrackbar("mean", win, 50, 100, update)
cv2.createTrackbar("var", win, 20, 100, update)

while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
