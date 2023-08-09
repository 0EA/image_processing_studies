import cv2
import matplotlib.pyplot as plt
import numpy as np


# resmi siyah beyaz olarak içe aktar
img = cv2.imread("odev1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# resim boyutu
print(img.shape)
img_width = img.shape[1]
img_height = img.shape[0]


# 4/5 oranında yeniden boyutlandırma
new_width = int(img_width * 4/5)
new_height = int(img_height * 4/5)
new_size = (new_width,new_height)
resized_img = cv2.resize(img, new_size)
print(resized_img.shape)

plt.figure()
plt.imshow(resized_img, cmap="gray")
plt.axis("off")
plt.show()


# yazı ekle
cv2.putText(img, "Kopke", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,10,10))
plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()


# threshold
_, thresh_img = cv2.threshold(img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.axis("off")
plt.show()
#inverse
_, thresh_img = cv2.threshold(img, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.axis("off")
plt.show()


# adaptive threshold
# 8 c sabiti, thresholda etkisi var
# 11 block size
thresh_img_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
plt.figure()
plt.imshow(thresh_img_2, cmap="gray")
plt.axis("off")
plt.title("Adaptive Threshold")
plt.show()


# gaussian blur
gb = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
plt.figure(), plt.imshow(gb, cmap="gray"), plt.axis("off"), plt.title("Gauss Blur"), plt.show()


# Laplacian gradient
laplacian = cv2.Laplacian(img, ddepth= cv2.CV_64F)
cv2.imshow("laplacian", laplacian)

# histogram
img_hist = cv2.imread("odev1.jpg")
img_hist = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image_hist = cv2.calcHist([img_hist], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.figure(), plt.plot(image_hist), plt.show()