import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 HSV 변환
image_bgr = cv2.imread('./Success_pic/blue2.jpg')
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# 초록색 범위 설정
lower_green = np.array([35, 40, 0])
upper_green = np.array([85, 255, 255])
green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

# 빨간색 범위 설정
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# 파란색 범위 설정
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)


# 두 개의 빨간색 마스크를 만듬
red_mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
red_mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# 초록색 마스크 적용
image_bgr_masked_green = cv2.bitwise_and(image_bgr, image_bgr, mask=green_mask)
image_rgb_green = cv2.cvtColor(image_bgr_masked_green, cv2.COLOR_BGR2RGB)

# 빨간색 마스크 적용
image_bgr_masked_red = cv2.bitwise_and(image_bgr, image_bgr, mask=red_mask)
image_rgb_red = cv2.cvtColor(image_bgr_masked_red, cv2.COLOR_BGR2RGB)

# 파란색 마스크 적용
image_bgr_masked_blue = cv2.bitwise_and(image_bgr, image_bgr, mask=blue_mask)
image_rgb_blue = cv2.cvtColor(image_bgr_masked_blue, cv2.COLOR_BGR2RGB)

# 이미지 출력
plt.imshow(image_rgb_green), plt.axis("off")
plt.show()

plt.imshow(image_rgb_red), plt.axis("off")
plt.show()

plt.imshow(image_rgb_blue), plt.axis("off")
plt.show()

#plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), plt.axis("off")
#plt.show()
