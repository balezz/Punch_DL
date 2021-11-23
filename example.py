import cv2
import numpy as np

from lib.utils import rotate


INPUT_SIZE = 300
SCALLING_FACTOR = 1 / 4

name = 'id0_jab_1'
video_path = f'./data/video/{name}.mp4'
keypoints = np.load(f'./data/keypoints/{name}.npy')
cap = cv2.VideoCapture(video_path)


while True:
    ret, frame = cap.read()

    frame_rotated = rotate(frame, 90)
    frame_resized = cv2.resize(frame_rotated, None, fx=SCALLING_FACTOR, fy=SCALLING_FACTOR)
    
    if not ret:
        break

    frame_flipped = cv2.flip(frame_resized, 1)

    cv2.imshow(name, frame_flipped)
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
