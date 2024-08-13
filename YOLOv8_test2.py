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

previous_y_values = None # previous frame y values to compare
first_point_threshold = None # difference in distance between keypoint 0 
second_point_threshold = None # difference in diqqqqqqstance between keypoint 5
falling_threshold = None # falling threshold
fallen_state = False # fallen state identifier
MIN_ELAPSED_TIME_THRESHOLD = 5
fall_start_time = None
elapsed_time_states = []

# VIDEO SAVING
VIDEO_FPS = 30
fall_alerted = False
video_frames_before = []
frozen_video_frames_before = []
video_frames_after = []
taking_video = False
clip_frames = []

def save_cropped_image(r):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 객체 경계 상자 좌표
    boxes = r.boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        # 사람을 감지했을 때
        x_min, y_min, x_max, y_max = boxes[0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        cropped_img = r.orig_img[y_min:y_max, x_min:x_max]

        # 객체 이미지 캡처
        img_save_path = f'./Success_pic/fallen_frame_{current_time}.png'
        cv2.imwrite(img_save_path, cropped_img)
        print(f'Fallen state detected. Image saved as {img_save_path}')

# GETTING THE Y VALUES OF THE PERSON
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
        falling_threshold = ((y_values_first_frame[len(y_values_first_frame)-1] - y_values_first_frame[0]) * 2/3) + 20 # Buffer (can be changed)
            
    print("Falling threshold:", falling_threshold)

    second_frame = next(results, None)
    if second_frame is not None:
        y_values_second_frame = frame_coordinates(second_frame)
        while(len(y_values_second_frame) < 6):
            second_frame = next(results, None)
            y_values_second_frame = frame_coordinates(second_frame)
    
    first_point_diff = abs(y_values_first_frame[0] - y_values_second_frame[0])
    second_point_diff = abs(y_values_first_frame[5] - y_values_second_frame[5])
    first_point_threshold = first_point_diff + 15 # Buffer (can be changed)
    second_point_threshold = second_point_diff + 15 # Buffer (can be changed)

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

    # This applies checking if it's a fall or laying down
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
                    #fallen_count+=1
                else:
                    fallen_state = True
                    taking_video = True
                    fall_start_time = time.time()
                    elapsed_time_states.append("Fallen")
                    print("Fallen")
                    print("STARTING TIME OF FALL:", fall_start_time)
                    print("states:", elapsed_time_states)

                    save_cropped_image(r)

                    """# 객체 이미지 캡처
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_save_path = f'./fallen_frame_{current_time}.png'
                    cv2.imwrite(img_save_path, r.orig_img)
                    print(f'Fallen state detected. Image saved as {img_save_path}')"""

        else:
            # If they are standing but 10 seconds have passed already after fall then we still want to see it
            # This handles it to keep taking the video
            if(fall_alerted):                
                taking_video = True
            else:
                # If they're just safe and stood up before the 10 seconds, then reset
                fallen_state = False
                fall_alerted = False
                taking_video = False
                frozen_video_frames_before.clear()
                video_frames_after.clear()
                
            fall_start_time = None
            elapsed_time_states.clear()
            print("Safe")

    # Update previous frame's values
    previous_y_values = y_values

def check_falling_time(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    global fallen_count
    # Perform subtraction only if fall_start_time is not None
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
            fallen_count+=1

    return fall_start_time, elapsed_time_states

def check_falling_time_out_of_frame(fall_start_time, elapsed_time_states):
    global fallen_state
    global taking_video
    global fall_alerted
    global duration_of_fall
    if fall_start_time is not None:
        # Perform subtraction only if fall_start_time is not None
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

    # Verify individual frames (optional)
    for frame in clip_frames:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame detected in clip_frames.")

    # Save the clip to a temporary file
    temp_file_path = tempfile.mktemp(suffix=".mp4")
    print(temp_file_path)
    out = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame.shape[1], frame.shape[0]))
    
    for frame in clip_frames:
        out.write(frame)

    out.release()

    print("File path:", temp_file_path)


# Replace 'your_video.mp4' with the path to your MP4 file
results = model(source='./test.mp4', show=True, conf=0.3, stream=True, save=False)
#results = model(source=0, show=True, conf=0.3, stream=True, save=True)


if results:
    first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
    for r in results:
        # Sliding window technique where we update the thresholds every 5 seconds
        if time.time() - start_time > 5 and fallen_state == False:
            first_point_threshold, second_point_threshold, start_time = get_starting_frames(results)
        # Update dynamic list with the latest frame
        print("Length of BEFORE:", len(video_frames_before))
        print("Length of FROZEN:", len(frozen_video_frames_before))
        print("Length of AFTER:", len(video_frames_after))
        print("Fall state:", fallen_state)
        print("Fall alerted:", fall_alerted)
        print("Taking video:", taking_video)

        # Add frames to start frames (before falling). This is continuous
        if len(video_frames_before) > 300:
            video_frames_before.pop(0)
        else:
            video_frames_before.append(r.orig_img) 

        # When a fall happens, taken video is true
        if taking_video:
            # Freeze before frames
            if(len(frozen_video_frames_before) == 0):
                frozen_video_frames_before = deepcopy(video_frames_before)
            # Add to after frames
            # If it's not 10 seconds after falling yet then keep adding frames
            if len(video_frames_after) <= 450:
                video_frames_after.append(r.orig_img)
            else:
                # When it's 10 seconds save video and send alert
                save_video_clip()
                taking_video = False
                video_frames_before.clear()
                video_frames_after.clear()
                frozen_video_frames_before.clear()
                fall_start_time = None
                elapsed_time_states.clear()
                fallen_state = False
        else:
            # If not taking video then update the frames before
            if len(video_frames_before) > 300:
                video_frames_before.pop(0)
            else:
                video_frames_before.append(r.orig_img) 

        y_values = frame_coordinates(r)
        # If 6 keypoints are showing then check for falling
        if len(y_values) >= 6:
            minimum = min(y_values)
            maximum = max(y_values)
            check_falling(y_values)
        else:
            # If less than 6 keypoints then check if they fell (No human is detected in this case)
            if fallen_state == True:
                elapsed_time_states.append("No human detected")
                fall_start_time, elapsed_time_states = check_falling_time_out_of_frame(fall_start_time, elapsed_time_states)
                print("No human detected.")
                print("states:", elapsed_time_states)

        cv2.imshow("Video Feed", r.orig_img)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
print("fallen_count ===== ",fallen_count)
cv2.destroyAllWindows()
