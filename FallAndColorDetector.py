import os
import time
import numpy as np
import cv2
import tkinter as tk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO
from copy import deepcopy
from datetime import datetime

class FallAndColorDetector(FileSystemEventHandler):
    def __init__(self, model_path, video_source, folder_to_watch, save_path="./Success_pic"):
        # Fall detection initialization
        self.model = YOLO(model_path)
        self.safe_count = 0
        self.fallen_count = 0
        self.previous_y_values = None
        self.first_point_threshold = None
        self.second_point_threshold = None
        self.falling_threshold = None
        self.fallen_state = False
        self.MIN_ELAPSED_TIME_THRESHOLD = 5
        self.fall_start_time = None
        self.elapsed_time_states = []
        self.fall_alerted = False
        self.video_frames_before = []
        self.frozen_video_frames_before = []
        self.video_frames_after = []
        self.taking_video = False
        self.clip_frames = []
        self.save_path = save_path
        self.VIDEO_FPS = 30
        self.results = self.model(source=video_source, show=True, conf=0.3, stream=True, save=False)

        # Color detection initialization
        self.color_info = {}
        self.folder_to_watch = folder_to_watch
        self.observer = Observer()
        self.observer.schedule(self, self.folder_to_watch, recursive=False)
        self.observer.start()

        # GUI initialization
        self.root = tk.Tk()
        self.root.title("Fall and Color Detection")
        self.text = tk.Text(self.root, height=20, width=60)
        self.text.pack()

    # Fall detection methods
    def get_starting_frames(self):
        start_time = time.time()
        first_frame = next(self.results, None)
        if first_frame is not None:
            y_values_first_frame = self.frame_coordinates(first_frame)
            while len(y_values_first_frame) < 6:
                first_frame = next(self.results, None)
                y_values_first_frame = self.frame_coordinates(first_frame)
            self.falling_threshold = ((y_values_first_frame[-1] - y_values_first_frame[0]) * 2/3) + 20
        
        print("Falling threshold:", self.falling_threshold)

        second_frame = next(self.results, None)
        if second_frame is not None:
            y_values_second_frame = self.frame_coordinates(second_frame)
            while len(y_values_second_frame) < 6:
                second_frame = next(self.results, None)
                y_values_second_frame = self.frame_coordinates(second_frame)
        
        first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
        second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
        self.first_point_threshold = first_point_diff + 15
        self.second_point_threshold = second_point_diff + 15

        print("First point threshold:", self.first_point_threshold)
        print("Second point threshold:", self.second_point_threshold)

        return self.first_point_threshold, self.second_point_threshold, start_time

    def frame_coordinates(self, frame):
        y_values_frame = [keypoint[1].numpy() for keypoint in frame.keypoints.xy[0] if keypoint[1].numpy() != 0]
        return y_values_frame

    def check_falling(self, y_values, r):
        if self.previous_y_values is not None and len(y_values) >= 6 and len(self.previous_y_values) >= 6:
            first_point_diff = abs(self.previous_y_values[0] - y_values[0])
            second_point_diff = abs(self.previous_y_values[5] - y_values[5])

            if (self.falling_threshold is not None) and (self.maximum - self.minimum <= self.falling_threshold):
                if (first_point_diff <= self.first_point_threshold) and (second_point_diff <= self.second_point_threshold):
                    print("Laying down")
                    if self.fallen_state:
                        self.elapsed_time_states.append("Laying down")
                        self.fall_start_time, self.elapsed_time_states = self.check_falling_time(self.fall_start_time, self.elapsed_time_states)
                        print("states:", self.elapsed_time_states)
                    else:
                        self.fall_start_time = None
                else:
                    if self.fallen_state:
                        self.elapsed_time_states.append("Fallen")
                        self.fall_start_time, self.elapsed_time_states = self.check_falling_time(self.fall_start_time, self.elapsed_time_states)
                        print("Fallen")
                        print("states:", self.elapsed_time_states)
                    else:
                        self.fallen_state = True
                        self.taking_video = True
                        self.fall_start_time = time.time()
                        self.elapsed_time_states.append("Fallen")
                        print("Fallen")
                        print("STARTING TIME OF FALL:", self.fall_start_time)
                        print("states:", self.elapsed_time_states)
                        self.save_cropped_image(r)
            else:
                if self.fall_alerted:                
                    self.taking_video = True
                else:
                    self.fallen_state = False
                    self.fall_alerted = False
                    self.taking_video = False
                    self.frozen_video_frames_before.clear()
                    self.video_frames_after.clear()
                self.fall_start_time = None
                self.elapsed_time_states.clear()
                print("Safe")
        self.previous_y_values = y_values

    def check_falling_time(self, fall_start_time, elapsed_time_states):
        if fall_start_time is not None:
            duration_of_fall = time.time() - fall_start_time
            print("Duration of fall:", duration_of_fall)
            if duration_of_fall >= self.MIN_ELAPSED_TIME_THRESHOLD:
                print("Elapsed time states:", elapsed_time_states)
                print("FALL ALERT!!!")
                self.fall_alerted = True
                self.taking_video = True
                fall_start_time = None
                elapsed_time_states.clear()
                self.fallen_state = False
                self.fallen_count += 1
        return fall_start_time, elapsed_time_states

    def save_cropped_image(self, r):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            x_min, y_min, x_max, y_max = map(int, boxes[0])
            cropped_img = r.orig_img[y_min:y_max, x_min:x_max]
            img_save_path = os.path.join(self.save_path, f'fallen_frame_{current_time}.png')
            cv2.imwrite(img_save_path, cropped_img)
            print(f'Fallen state detected. Image saved as {img_save_path}')

            # Immediately process the saved image for color detection
            self.process_image(img_save_path)

    def save_video_clip(self):
        self.clip_frames = self.frozen_video_frames_before + self.video_frames_after
        if not self.clip_frames:
            print("No frames to save.")
            return
        temp_file_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), self.VIDEO_FPS, (self.clip_frames[0].shape[1], self.clip_frames[0].shape[0]))
        for frame in self.clip_frames:
            out.write(frame)
        out.release()
        print("File path:", temp_file_path)

    def run_fall_detection(self):
        if self.results:
            first_point_threshold, second_point_threshold, start_time = self.get_starting_frames()
            for r in self.results:
                if time.time() - start_time > 5 and self.fallen_state == False:
                    first_point_threshold, second_point_threshold, start_time = self.get_starting_frames()
                print("Length of BEFORE:", len(self.video_frames_before))
                print("Length of FROZEN:", len(self.frozen_video_frames_before))
                print("Length of AFTER:", len(self.video_frames_after))
                print("Fall state:", self.fallen_state)
                print("Fall alerted:", self.fall_alerted)
                print("Taking video:", self.taking_video)
                if len(self.video_frames_before) > 300:
                    self.video_frames_before.pop(0)
                else:
                    self.video_frames_before.append(r.orig_img)
                if self.taking_video:
                    if len(self.frozen_video_frames_before) == 0:
                        self.frozen_video_frames_before = deepcopy(self.video_frames_before)
                    if len(self.video_frames_after) <= 450:
                        self.video_frames_after.append(r.orig_img)
                    else:
                        #self.save_video_clip()
                        self.taking_video = False
                        self.video_frames_before.clear()
                        self.video_frames_after.clear()
                        self.frozen_video_frames_before.clear()
                        self.fall_start_time = None
                        self.elapsed_time_states.clear()
                        self.fallen_state = False
                y_values = self.frame_coordinates(r)
                if len(y_values) >= 6:
                    self.minimum = min(y_values)
                    self.maximum = max(y_values)
                    self.check_falling(y_values, r)
                cv2.imshow("Video Feed", r.orig_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

    # Color detection methods
    def on_created(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            print(f"File created: {event.src_path}")
            self.process_image(event.src_path)

    def process_image(self, path):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        # 파일 접근 권한 문제를 처리하기 위한 대기 및 재시도
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"Failed to read the image. Skipping file: {path}")
                    return
                break
            except PermissionError:
                print(f"Attempt {attempt + 1} of {max_attempts}: Permission denied for file {path}. Retrying...")
                time.sleep(1)
        else:
            print(f"Failed to open the file after {max_attempts} attempts: {path}")
            return

        # 이후 코드 계속 진행
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
            result, mask = self.detect_color(img, lower_bounds, upper_bounds)
            color_proportion = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            if color_proportion > threshold:

            # 해당 부분에 Flask 연동 될 수 있게 수정하면 됨(color_name => 색상, color_proportion => 색상 비율로 설정)

                detected_colors.append((color_name, color_proportion))
                print(f"Detected {color_name} with proportion {round(color_proportion, 2)} in {path}")
                color_found = True

        if not color_found:
            detected_colors.append(("Undefined", 0))
            print(f"No significant color detected in {path}, marked as Undefined.")
        self.color_info[file_creation_time] = detected_colors
        print("\nCurrent color_info dictionary:")
        print(self.color_info)
        self.display_color_info()

    def detect_color(self, img, lower_bounds, upper_bounds):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]
        mask = np.bitwise_or.reduce(masks)
        result = cv2.bitwise_and(img, img, mask=mask)
        return result, mask

    def display_color_info(self):
        self.text.delete(1.0, tk.END)
        for timestamp, colors in self.color_info.items():
            self.text.insert(tk.END, f"Timestamp: {timestamp}\n")
            for color, proportion in colors:
                self.text.insert(tk.END, f"Color: {color}, Proportion: {proportion:.2f}\n")
            self.text.insert(tk.END, "\n")
        self.root.update()

    def run(self):
        self.root.after(0, self.run_fall_detection)
        self.root.mainloop()
        self.observer.join()

if __name__ == '__main__':
    detector = FallAndColorDetector(model_path='yolo models/yolov8s-pose.pt', video_source='./test.mp4', folder_to_watch='./Success_pic')
    detector.run()
