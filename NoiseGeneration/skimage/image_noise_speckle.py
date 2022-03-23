import sys

import cv2
import skimage

win = "speckle"

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

cv2.imshow("image", img)


def update(_):
    mean = cv2.getTrackbarPos("mean", win) / 100.0 - 0.5
    var = cv2.getTrackbarPos("var", win) / 2000.0
    noised = skimage.util.random_noise(
        img, seed=42, clip=True, mean=mean, var=var, mode=win
    )
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
