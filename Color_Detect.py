import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_red(img):
    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 정의 (두 범위를 사용해 원형으로 표현되는 빨간색을 모두 포함)
    lower_red1 = np.array([0, 100, 100], dtype="uint8")
    upper_red1 = np.array([10, 255, 255], dtype="uint8")
    lower_red2 = np.array([160, 100, 100], dtype="uint8")
    upper_red2 = np.array([180, 255, 255], dtype="uint8")
    
    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask

# 이미지 파일 경로
image_paths = [
    "./Success_pic/fallen_frame_20240812_112747.png",
    "./Success_pic/images.jpg",
    "./Success_pic/AKR20230316068500805_19_i_P4.jpg",
    "./Success_pic/black_check.jpeg",
    "./Success_pic/red_vest.jpg"
]

for path in image_paths:
    img = cv2.imread(path)
    result, mask = detect_red(img)
    
    # 원본 이미지와 결과 이미지를 가로로 합침
    result_combined = np.hstack([img, result])
    
    # 결과 이미지와 마스크를 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result_combined, cv2.COLOR_BGR2RGB))
    plt.title('Original and Red Detected')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Red Mask')
    plt.axis('off')
    
    plt.show()

    # 빨간색 비율 계산
    red_proportion = round(np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]), 2)
    print(f'Proportion of red in {path}: {red_proportion}')
