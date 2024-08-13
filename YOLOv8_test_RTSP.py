from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
from copy import deepcopy
from datetime import datetime

model = YOLO('yolo models/yolov8s-pose.pt')

safe_count = 0
fallen_count = 0

previous_y_values = None
first_point_threshold = None
second_point_threshold = None
falling_threshold = None
fallen_state = False
MIN_ELAPSED_TIME_THRESHOLD = 5
fall_start_time = None
elapsed_time_states = []

VIDEO_FPS = 30
fall_alerted = False
video_frames_before = []
frozen_video_frames_before = []
video_frames_after = []
taking_video = False
clip_frames = []

# 최대 재시도 횟수 및 대기 시간 설정
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

def save_cropped_image(r):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    boxes = r.boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        x_min, y_min, x_max, y_max = boxes[0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        cropped_img = r.orig_img[y_min:y_max, x_min:x_max]
        img_save_path = f'./Success_pic/fallen_frame_{current_time}.png'
        cv2.imwrite(img_save_path, cropped_img)
        print(f'Fallen state detected. Image saved as {img_save_path}')

def frame_coordinates(frame):
    y_values_frame = [keypoint[1].numpy() for keypoint in frame.keypoints.xy[0] if keypoint[1].numpy() != 0]
    return y_values_frame

def get_starting_frames(results):
    global first_point_threshold
    global second_point_threshold
    global falling_threshold

    start_time = time.time()

    first_frame = next(results, None)
    if first_frame is not None:
        y_values_first_frame = frame_coordinates(first_frame)
        while(len(y_values_first_frame) < 6):
            first_frame = next(results, None)
            y_values_first_frame = frame_coordinates(first_frame)
        falling_threshold = ((y_values_first_frame[len(y_values_first_frame)-1] - y_values_first_frame[0]) * 2/3) + 20 
            
    print("Falling threshold:", falling_threshold)

    second_frame = next(results, None)
    if second_frame is not None:
        y_values_second_frame = frame_coordinates(second_frame)
        while(len(y_values_second_frame) < 6):
            second_frame = next(results, None)
            y_values_second_frame = frame_coordinates(second_frame)
    
    first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
    second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
    first_point_threshold = first_point_diff + 15
    second_point_threshold = second_point_diff + 15

    print("First point threshold:", first_point_threshold)
    print("Second point threshold:", second_point_threshold)

    return first_point_threshold, second_point_threshold, start_time

def check_falling(y_values):
    global previous_y_values
    global fallen_state
    global minimum
    global maximum
    global fall_start_time
    global elapsed_time_states
    global taking_video
    global fall_alerted
    global video_frames_after
    global fallen_count

    if previous_y_values is not None and len(y_values) >= 6 and len(previous_y_values) >= 6:
        first_point_diff = abs(previous_y_values[0] - y_values[0])
        second_point_diff = abs(previous_y_values[5] - y_values[5])

        if (falling_threshold is not None) and (maximum - minimum <= falling_threshold):
            if (first_point_diff <= first_point_threshold) and (second_point_diff <= second_point_threshold):
                print("Laying down")
                if fallen_state:
                    elapsed_time_states.append("Laying down")
                    fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
                    print("states:", elapsed_time_states)
                    
                else:
                    fall_start_time = None
            else:
                if fallen_state:
                    elapsed_time_states.append("Fallen")
                    fall_start_time, elapsed_time_states = check_falling_time(fall_start_time, elapsed_time_states)
                    print("Fallen")
                    print("states:", elapsed_time_states)
                else:
                    fallen_state = True
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("Fallen")
                    print("STARTING TIME OF FALL:", fall_start_time)
                    print("states:", elapsed_time_states)

                    save_cropped_image(r)

        else:
            if(fall_alerted):                
                taking_video = True
            else:
                fallen_state = False
                fall_alerted = False
                taking_video = False
                frozen_video_frames_before.clear()
                video_frames_after.clear()
                
            fall_start_time = None
            elapsed_time_states.clear()
            print("Safe")

    previous_y_values = y_values

def check_falling_time(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    global fallen_count
    if fall_start_time is not None:
        duration_of_fall = time.time() - fall_start_time
        print("Duration of fall:", duration_of_fall)
        if duration_of_fall >= MIN_ELAPSED_TIME_THRESHOLD:
            print("Elapsed time states:", elapsed_time_states)
            print("FALL ALERT!!!")
            fall_alerted = True
            taking_video = True
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False
            fallen_count += 1

    return fall_start_time, elapsed_time_states

def check_falling_time_out_of_frame(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    if fall_start_time is not None:
        duration_of_fall = time.time() - fall_start_time
        print("Duration of fall:", duration_of_fall)
        if duration_of_fall >= MIN_ELAPSED_TIME_THRESHOLD:
            print("Elapsed time states:", elapsed_time_states)
            print("FALL ALERT!!!")
            while len(video_frames_after) <= 450:
                video_frames_after.append(r.orig_img)
            save_video_clip()
            taking_video = False
            video_frames_before.clear()
            video_frames_after.clear()
            frozen_video_frames_before.clear()
            fall_start_time = None
            elapsed_time_states.clear()
            fallen_state = False
    return fall_start_time, elapsed_time_states

def save_video_clip():
    global clip_frames
    clip_frames = frozen_video_frames_before + video_frames_after

    if not clip_frames:
        print("No frames to save.")
        return

    for frame in clip_frames:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame detected in clip_frames.")

    temp_file_path = tempfile.mktemp(suffix=".mp4")
    print(temp_file_path)
    out = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame.shape[1], frame.shape[0]))
    
    for frame in clip_frames:
        out.write(frame)

    out.release()

    print("File path:", temp_file_path)


def process_rtsp_stream(rtsp_url):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"Attempting to connect to RTSP stream. Try {retries + 1}/{MAX_RETRIES}")
            results = model(source=rtsp_url, show=True, conf=0.3, stream=True, save=False)
            if not results:
                raise ValueError("No results from the RTSP stream.")

            if results:
                first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
                for r in results:
                    if time.time() - start_time > 5 and fallen_state == False:
                        first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
                    print("Length of BEFORE:", len(video_frames_before))
                    print("Length of FROZEN:", len(frozen_video_frames_before))
                    print("Length of AFTER:", len(video_frames_after))
                    print("Fall state:", fallen_state)
                    print("Fall alerted:", fall_alerted)
                    print("Taking video:", taking_video)

                    if len(video_frames_before) > 300:
                        video_frames_before.pop(0)
                    else:
                        video_frames_before.append(r.orig_img) 

                    if taking_video:
                        if(len(frozen_video_frames_before) == 0):
                            frozen_video_frames_before = deepcopy(video_frames_before)
                        if len(video_frames_after) <= 450:
                            video_frames_after.append(r.orig_img)
                        else:
                            save_video_clip()
                            taking_video = False
                            video_frames_before.clear()
                            video_frames_after.clear()
                            frozen_video_frames_before.clear()
                            fall_start_time = None
                            elapsed_time_states.clear()
                            fallen_state = False
                    else:
                        if len(video_frames_before) > 300:
                            video_frames_before.pop(0)
                        else:
                            video_frames_before.append(r.orig_img) 

                    y_values = frame_coordinates(r)
                    if len(y_values) >= 6:
                        minimum = min(y_values)
                        maximum = max(y_values)
                        check_falling(y_values)
                    else:
                        if fallen_state == True:
                            elapsed_time_states.append("No human detected")
                            fall_start_time, elapsed_time_states = check_falling_time_out_of_frame(fall_start_time, elapsed_time_states)
                            print("No human detected.")
                            print("states:", elapsed_time_states)

                    cv2.imshow("Video Feed", r.orig_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            print("fallen_count ===== ", fallen_count)
            cv2.destroyAllWindows()
            break

        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Maximum retries reached. Exiting...")
                break


# 실행 부분
rtsp_url = "rtsp://210.99.70.120:1935/live/cctv001.stream"
process_rtsp_stream(rtsp_url)
