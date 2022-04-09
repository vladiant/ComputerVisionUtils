import sys

import cv2
import numpy as np

win = "localvar"


def localvar(image, local_vars, seed):
    rng = np.random.default_rng(seed)
    # Ensure local variance input is correct
    if (local_vars <= 0).any():
        raise ValueError("All values of `local_vars` must be > 0.")

    # Safe shortcut usage broadcasts 'local_vars' as a ufunc
    out = image + rng.normal(0, local_vars ** 0.5)

    out = np.clip(out, 0.0, 1.0)
    return out


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

# Convert to float
img = np.float32(img) / 255.0

cv2.imshow("image", img)


def update(_):
    # local_vars - defining the local variance at every image point
    noised = localvar(img, img / 50.0, 42)
    cv2.imshow(win, noised)


cv2.namedWindow("image")
cv2.namedWindow(win)


while True:
    update(None)
    ch = cv2.waitKey()
    if ch == 27:
        break
