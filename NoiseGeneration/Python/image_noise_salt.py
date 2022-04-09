import sys

import cv2
import numpy as np

win = "salt"


def bernoulli(p, shape, random_state):
    return random_state.random(shape) <= p


def salt(image, amount, seed):
    rng = np.random.default_rng(seed)
    out = image.copy()
    p = amount
    flipped = bernoulli(p, image.shape, rng)
    out[flipped] = 1.0

    return out


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

# Convert to float
img = np.float32(img) / 255.0

cv2.imshow("image", img)


def update(_):
    amount = cv2.getTrackbarPos("amount", win) / 100.0
    noised = salt(img, amount, 42)
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


cv2.createTrackbar("amount", win, 50, 100, update)

while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
