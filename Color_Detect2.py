import cv2
import numpy as np
import matplotlib.pyplot as plt

#green색 mask적용
image_bgr = cv2.imread('./Success_pic/green.jpg') # 이미지 로드
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환
lower_green = np.array([35,40,0]) # HSV에서 색의 값 범위 정의 
upper_green = np.array([85,255,255]) # hue(색상), saturation(채도), value(명도)
mask = cv2.inRange(image_hsv, lower_green, upper_green) # 마스크를 만듬 

image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask) # 이미지에 마스크를 적용
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 변환

#plt.imshow(mask), plt.axis("off") # 이미지 출력
#plt.show()

plt.imshow(image_rgb), plt.axis("off") # 이미지 출력
plt.show()