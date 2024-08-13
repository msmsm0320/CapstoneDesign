import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 색상 정보를 저장할 사전
color_info = {}

class Watcher:
    def __init__(self, folder_to_watch):
        self.folder_to_watch = folder_to_watch
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.folder_to_watch, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            print(f"File created: {event.src_path}")
            process_image(event.src_path)

def process_image(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read the image. Skipping file: {path}")
        return

    # 사진의 생성 시간을 키로 사용
    file_creation_time = time.ctime(os.path.getctime(path))

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

    threshold = 0.1
    detected_colors = []

    print(f"Processing image: {path}")

    color_found = False
    
    for color_name, bounds in color_ranges.items():
        lower_bounds, upper_bounds = zip(*bounds)
        result, mask = detect_color(img, lower_bounds, upper_bounds)

        color_proportion = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

        if color_proportion > threshold:
            detected_colors.append((color_name, color_proportion))
            print(f"Detected {color_name} with proportion {round(color_proportion, 2)} in {path}")

            result_combined = np.hstack([img, result])

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
            
            color_found = True

    # 탐지된 색상이 없으면 Undefined로 저장
    if not color_found:
        detected_colors.append(("Undefined", 0))
        print(f"No significant color detected in {path}, marked as Undefined.")

    # 색상 정보를 사전에 저장
    color_info[file_creation_time] = detected_colors

    # color_info 사전 전체를 출력
    print("\nCurrent color_info dictionary:")
    print(color_info)

def detect_color(img, lower_bounds, upper_bounds):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]
    mask = np.bitwise_or.reduce(masks)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result, mask

if __name__ == '__main__':
    folder_to_watch = "./Success_pic"  # 감시할 폴더 경로
    watcher = Watcher(folder_to_watch)
    watcher.run()
