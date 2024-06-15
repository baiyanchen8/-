import cv2
import numpy as np

def adjust_levels(image, lower, upper):
    img_min = np.min(image)
    img_max = np.max(image)
    adjusted = (image - img_min) / (img_max - img_min) * (upper - lower) + lower
    return adjusted

def to_cmy(image):
    cmy = 255 - image
    return cmy[:, :, 0], cmy[:, :, 1], cmy[:, :, 2]

def halftone(image):
    image = image.astype(np.float32)
    rows, cols = image.shape

    for y in range(rows):
        for x in range(cols):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < cols:
                image[y, x + 1] += quant_error * 7 / 16
            if y + 1 < rows:
                image[y + 1, x] += quant_error * 5 / 16
                if x - 1 >= 0:
                    image[y + 1, x - 1] += quant_error * 3 / 16
                if x + 1 < cols:
                    image[y + 1, x + 1] += quant_error * 1 / 16

    return image

def error_diffusion(image, oimage, y, x):
    old_pixel = (oimage[y, x])
    new_pixel = (image[y, x])
    error = old_pixel - new_pixel
    rows, cols = len(image), len(image[0])
    value = 255
    n = 3
    if x + 1 < cols:
        image[y, x + 1] += (error * 7 / 16)
        image[y, x + 1] = np.clip(image[y, x + 1], 0, 255)
    if y + 1 < rows:
        image[y + 1, x] += (error * 5 / 16)
        image[y + 1, x] = np.clip(image[y + 1, x], 0, 255)
        if x - 1 >= 0:
            image[y + 1, x - 1] += (error * 3 / 16)
            image[y + 1, x - 1] = np.clip(image[y + 1, x - 1], 0, 255)
        if x + 1 < cols:
            image[y + 1, x + 1] += (error * 1 / 16)
            image[y + 1, x + 1] = np.clip(image[y + 1, x + 1], 0, 255)

    return image

def cmy_to_rgb(C, M, Y):
    R = 255 - C
    G = 255 - M
    B = 255 - Y
    return np.dstack([R, G, B])

def merge_image(C, M, Y):
    return np.dstack((C, M, Y))

def process_images(C1, C2, S, CT1, CT2, ST1, ST2):
    T = (CT1 + CT2) / 2

    C1_adj = adjust_levels(C1, CT1, CT2)
    C2_adj = adjust_levels(C2, CT1, CT2)
    S_adj = adjust_levels(S, ST1, ST2)

    C1C, C1M, C1Y = to_cmy(C1_adj)
    C2C, C2M, C2Y = to_cmy(C2_adj)
    SC, SM, SY = to_cmy(S_adj)
    OC1C, OC1M, OC1Y = C1C.copy(), C1M.copy(), C1Y.copy()
    OC2C, OC2M, OC2Y = C2C.copy(), C2M.copy(), C2Y.copy()

    SC = halftone(SC)
    SM = halftone(SM)
    SY = halftone(SY)



    for channel1, channel2, S_channel, OOasd1, OOasd2 in [(C1C, C2C, SC, OC1C, OC2C), (C1M, C2M, SM, OC1M, OC2M), (C1Y, C2Y, SY, OC1Y, OC2Y)]:
        for i in range(SC.shape[0]):
            for j in range(SC.shape[1]):
                if S_channel[i, j] == 255:
                    channel1[i, j] = 255
                    channel2[i, j] = 255
                elif S_channel[i, j] == 0:
                    if channel1[i, j] >= T and channel2[i, j] >= T:
                        if channel1[i, j] > channel2[i, j]:
                            channel1[i, j], channel2[i, j] = 255, 0
                        else:
                            channel1[i, j], channel2[i, j] = 0, 255
                    elif channel1[i, j] >= T:
                        channel1[i, j], channel2[i, j] = 255, 0
                    elif channel2[i, j] >= T:
                        channel1[i, j], channel2[i, j] = 0, 255
                    else:
                        channel1[i, j], channel2[i, j] = 0, 0

                channel1 = error_diffusion(channel1, OOasd1, i, j)
                channel2 = error_diffusion(channel2, OOasd2, i, j)


    Share1 = merge_image(C1C, C1M, C1Y)
    Share2 = merge_image(C2C, C2M, C2Y)

    return Share1, Share2

# 重建影像
def to_cmy_2(image):
    return 255 - image

######################################################################### 

# 測試範例
C1 = cv2.imread('./resource_image/C1.png')
C2 = cv2.imread('./resource_image/C2.png')
S = cv2.imread('./resource_image/S.png')

C1 = C1.astype(np.float32)
C2 = C2.astype(np.float32)
S = S.astype(np.float32)

CT1, CT2 = 15, 200
ST1, ST2 = 150, 180


Share1, Share2 = process_images(C1, C2, S, CT1, CT2, ST1, ST2)

Share1 = Share1.astype(np.uint8)
Share2 = Share2.astype(np.uint8)

C1 = to_cmy_2(C1)
C2 = to_cmy_2(C2)

cv2.imwrite('result_image/Share1.png', Share1)
cv2.imwrite('result_image/Share2.png', Share2)
def cmy_to_rgb_test(C, M, Y):
    R = 255 - C
    G = 255 - M
    B = 255 - Y
    return cv2.merge([R, G, B])

temp1 = cv2.imread('result_image/Share1.png')
temp2 = cv2.imread('result_image/Share2.png')

cv2.imwrite('result_image/Share1.png', cmy_to_rgb_test(temp1[:,:,0], temp1[:,:,1], temp1[:,:,2]))
cv2.imwrite('result_image/Share2.png', cmy_to_rgb_test(temp2[:,:,0], temp2[:,:,1], temp2[:,:,2]))