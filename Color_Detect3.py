import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def detect_color(img, lower_bounds, upper_bounds):
    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 여러 범위에 대한 마스크 생성
    masks = [cv2.inRange(hsv, lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]
    
    # 모든 마스크를 결합
    mask = np.bitwise_or.reduce(masks)
    
    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask

# 색상 범위 정의 (빨간색, 파란색, 녹색)
color_ranges = {
    "red": [
        (np.array([0, 100, 100], dtype="uint8"), np.array([10, 255, 255], dtype="uint8")),
        (np.array([160, 100, 100], dtype="uint8"), np.array([179, 255, 255], dtype="uint8"))
    ],
    "blue": [
        (np.array([100, 100, 100], dtype="uint8"), np.array([140, 255, 255], dtype="uint8"))
    ],
    "green": [
        (np.array([40, 100, 100], dtype="uint8"), np.array([70, 255, 255], dtype="uint8"))
    ]
}

# 이미지 파일 경로
image_paths = [
    "./Success_pic/fallen_frame_20240812_112747.png",
    "./Success_pic/images.jpg",
    "./Success_pic/AKR20230316068500805_19_i_P4.jpg",
    "./Success_pic/black_check.jpeg",
    "./Success_pic/red_vest.jpg",
    "./Success_pic/blue1.jpg",
    "./Success_pic/blue2.jpg",
    "./Success_pic/green.jpg",
    "./Success_pic/orange1.jpg"
]

# 임계값 설정 (예: 0.1은 10%)
threshold = 0.1

# 결과를 저장할 큐
detected_colors = deque()

for path in image_paths:
    img = cv2.imread(path)
    print(f"Processing image: {path}")
    
    color_found = False  # 임계값을 넘는 색상이 있는지 확인하기 위한 플래그
    
    for color_name, bounds in color_ranges.items():
        lower_bounds, upper_bounds = zip(*bounds)
        result, mask = detect_color(img, lower_bounds, upper_bounds)
        
        # 색상 비율 계산
        color_proportion = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        # 임계값 초과 시 색상 저장 및 표시
        if color_proportion > threshold:
            detected_colors.append((path, color_name, color_proportion))
            print(f"Detected {color_name} with proportion {round(color_proportion, 2)} in {path}\n")
            
            # 원본 이미지와 결과 이미지를 가로로 합침
            result_combined = np.hstack([img, result])
            
            # 결과 이미지와 마스크를 시각화
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(result_combined, cv2.COLOR_BGR2RGB))
            plt.title(f'Original and {color_name.capitalize()} Detected')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f'{color_name.capitalize()} Mask')
            plt.axis('off')
            
            plt.show()
            
            color_found = True  # 임계값을 넘는 색상이 있음을 표시
    
    # 임계값을 넘는 색상이 없을 경우
    if not color_found:
        detected_colors.append((path, "Undefined", 0))
        print(f"No significant color detected in {path}, marked as Undefined.\n")

# 큐에 저장된 결과 출력
print("\nSummary of detected colors above threshold:")
while detected_colors:
    path, color_name, proportion = detected_colors.popleft()
    if color_name == "Undefined":
        print(f"Image: {path} - No significant color detected (Undefined)")
    else:
        print(f"Image: {path}, Color: {color_name}, Proportion: {round(proportion, 2)}")
