import skimage
import skimage.io
import matplotlib.pyplot as plt

img_path = "Lenna.png"
img = skimage.io.imread(img_path) / 255.0


def plotnoise(img, mode, r, c, i):
    plt.subplot(r, c, i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg)
    else:
        plt.imshow(img)
    plt.title(mode)
    plt.axis("off")


plt.figure(figsize=(18, 24))
r = 4
c = 2
plotnoise(img, "gaussian", r, c, 1)
plotnoise(img, "localvar", r, c, 2)
plotnoise(img, "poisson", r, c, 3)
plotnoise(img, "salt", r, c, 4)
plotnoise(img, "pepper", r, c, 5)
plotnoise(img, "s&p", r, c, 6)
plotnoise(img, "speckle", r, c, 7)
plotnoise(img, None, r, c, 8)
plt.show()
