import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_red(img):
    red_range = ([35, 40, 0], [85, 255, 255])
    red_lower_bound = np.array([35, 40, 0], dtype="uint8")
    red_upper_bound = np.array([85, 255, 255], dtype="uint8")
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(cvt, lowerb=red_lower_bound, upperb=red_upper_bound)
    result = cv2.bitwise_and(cvt, cvt, mask=mask)
    return result, mask

# 각 이미지 파일 경로
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
    red_proportion = round(len(result[result != 0]) / (result.shape[0] * result.shape[1] * 3), 2)
    print(f'Proportion of red in {path}: {red_proportion}')


"""import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_red(img):
    red_range = ([10, 10, 70], [80, 80, 255])
    lower = np.array(red_range[0], dtype = "uint8")
    upper = np.array(red_range[1], dtype = "uint8")
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(cvt, lowerb = lower, upperb = upper)
    result = cv2.bitwise_and(cvt, cvt, mask = mask)
    return result

image_path = "./Success_pic/black_check.jpeg"
img = cv2.imread(image_path)
result = detect_red(img)
result = np.hstack([img, result])
plt.imshow(result)
print('proportion of red : {}'.format(round(len(result[result!=0])/(result.shape[0]*result.shape[1]*3),2)))
"""