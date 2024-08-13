import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# RTSP 스트림 URL 리스트
rtsp_urls = [
    "rtsp://210.99.70.120:1935/live/cctv001.stream",
    "rtsp://210.99.70.120:1935/live/cctv002.stream",
    "rtsp://210.99.70.120:1935/live/cctv003.stream",
    "rtsp://210.99.70.120:1935/live/cctv004.stream"
]

# 각 스트림에 대해 OpenCV 비디오 캡처 객체 생성
caps = [cv2.VideoCapture(url) for url in rtsp_urls]

def update_frame(idx, lbl_video):
    cap = caps[idx]
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    # 25ms 후에 다시 update_frame 함수 호출
    lbl_video.after(25, update_frame, idx, lbl_video)

# tkinter 창 설정
root = tk.Tk()
root.title("Multiple RTSP Video Streams")

# 2x2 그리드 레이아웃을 위한 설정
labels = []
for i in range(4):
    lbl = tk.Label(root)
    lbl.grid(row=i//2, column=i%2)  # 2x2 그리드 레이아웃
    labels.append(lbl)

# 각 스트림에 대해 업데이트 시작
for i in range(4):
    root.after(0, update_frame, i, labels[i])

# tkinter 메인 루프 시작
root.mainloop()

# 종료 시 비디오 캡처 객체 해제
for cap in caps:
    cap.release()
