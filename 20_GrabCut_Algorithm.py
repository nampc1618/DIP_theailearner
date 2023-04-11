# This is algorithm for foreground extraction

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('.//imageTest//messi.jpg')
# Create a 0's mask
mask = np.zeros(img.shape[:2], np.uint8)
# Create 2 arrays for background and foreground model
bgdModel = np.zeros((1, 65),np.float64)
fgdModel = np.zeros((1, 65),np.float64)

rect = (210,40,195,385)
mask, bgdModel, fgdModel, cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_seg= img*mask2[:,:,np.newaxis]
#cv2.imwrite(".//imageTest//messi_mark.jpg",img_seg)
plt.imshow(mask2, cmap='gray')
plt.show()
plt.imshow(img_seg)
plt.show()
# img_rect = cv2.rectangle(img, rect, (0, 0, 255), 2)
# cv2.imshow("Messi", img_rect)

# Load the marked image
img_mark = cv2.imread('.//imageTest//messi_mark.jpg')
# Subtract to obtain the mask
mask_dif = cv2.subtract(img_mark, img_seg)
# Convert the mask to grey and threshold it
mask_grey = cv2.cvtColor(mask_dif, cv2.COLOR_BGR2GRAY)
ret, mask1 = cv2.threshold(mask_grey, 200, 255, 0)
plt.imshow(mask1, cmap='gray')
plt.show()

mask[mask1==255] = 1
mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask_final = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_out = img*mask_final[:,:,np.newaxis]
cv2.imwrite(".//imageTest//messi_removebg.jpg",img_out)

cv2.imshow("out",img_out)
plt.imshow(img_out)
plt.show()
cv2.waitKey()