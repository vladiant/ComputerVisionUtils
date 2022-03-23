import sys

import cv2
import skimage

win = "salt"

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

cv2.imshow("image", img)


def update(_):
    amount = cv2.getTrackbarPos("amount", win) / 100.0
    noised = skimage.util.random_noise(img, seed=42, clip=True, amount=amount, mode=win)
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


cv2.createTrackbar("amount", win, 50, 100, update)

while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
