import cv2
import numpy as np
def to_cmy_2(image):
    return 255 - image

C1 = cv2.imread('./result_image/Share1.png')
C2 = cv2.imread('./result_image/Share2.png')



c3 = np.zeros(C2.shape, dtype=np.uint8)

for i in range(C2.shape[0]):
    for j in range(C2.shape[1]):
        c3[i, j] = np.clip(C1[i, j] + C2[i, j], 0, 255)


cv2.imwrite('./result_image/reconstruct.png', c3)


