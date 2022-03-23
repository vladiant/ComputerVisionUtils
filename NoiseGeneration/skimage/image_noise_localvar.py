import sys

import cv2
import skimage

win = "localvar"

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

cv2.imshow("image", img)


def update(_):
    # local_vars - defining the local variance at every image point
    noised = skimage.util.random_noise(
        img, seed=42, clip=True, local_vars=img / 10000.0, mode=win
    )
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
