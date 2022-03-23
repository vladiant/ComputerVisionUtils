import sys

import cv2
import skimage

win = "s&p"

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

cv2.imshow("image", img)


def update(_):
    amount = cv2.getTrackbarPos("amount", win) / 100.0
    salt_vs_pepper = cv2.getTrackbarPos("salt_vs_pepper", win) / 100.0
    noised = skimage.util.random_noise(
        img, seed=42, clip=True, amount=amount, salt_vs_pepper=salt_vs_pepper, mode=win
    )
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


cv2.createTrackbar("amount", win, 50, 100, update)
cv2.createTrackbar("salt_vs_pepper", win, 50, 100, update)

while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
