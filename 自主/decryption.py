import cv2
import numpy as np

# 讀取兩張圖片
C1 = cv2.imread('./result_image/Share1.png')
C2 = cv2.imread('./result_image/Share2.png')

# 確保兩張圖片大小一致
C1 = cv2.resize(C1, (C2.shape[1], C2.shape[0]))

# 新建一個空白的影像來存放疊合後的結果
c3 = np.zeros(C2.shape, dtype=np.uint8)

# 使用 OR 運算將兩張圖片疊合
for i in range(C2.shape[0]):
    for j in range(C2.shape[1]):
        c3[i, j] = C1[i, j] | C2[i, j]  # 對每個像素進行逐位OR操作

# 儲存疊合後的影像
cv2.imwrite('./result_image/reconstruct.png', c3)
