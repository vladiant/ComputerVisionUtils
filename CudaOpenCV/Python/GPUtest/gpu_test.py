# https://learnopencv.com/getting-started-opencv-cuda-module/

import sys
import cv2

if len(sys.argv) < 2:
    print(f"Program usage: {sys.argv[0]} image_filename")
    exit(1)

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

src = cv2.cuda_GpuMat()
src.upload(img)

clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst = clahe.apply(src, cv2.cuda_Stream.Null())

result = dst.download()

cv2.imshow("result", result)
cv2.waitKey(0)
