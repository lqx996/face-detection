import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import time
# time_start=time.time()

thresh = 0.8
scales = [1024, 1980]
count = 1
gpuid = 1
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
videopath = "/workspace/liu/remote_workspace/video2019/houcheting2.mp4"
video_capture = cv2.VideoCapture(videopath)
corpbbox = None
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = video_capture.get(cv2.CAP_PROP_FPS)
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print (size)
out = cv2.VideoWriter('/workspace/liu/remote_workspace/output/houcheting4.mp4',fourcc, 30, size,True)
frame_index=0
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(size[0:2])
print(im_size_min)
im_size_max = np.max(size[0:2])
im_scale = float(target_size) / float(im_size_min)
print(im_scale)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
scales = [im_scale]
# time_start=time.time()
while True:

    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()
    interval = 5
    time_start = time.time()

    if ret:
       print ("---> %d:" % frame_index, ret)
            # img_corner = np.array(cv2.resize(frame, (640, 360)))
       if frame_index % interval == 0:
            img = np.array(frame)
            flip = False
            faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
       for i in range(faces.shape[0]):
              # print('score', faces[i][4])
              box = faces[i].astype(np.int)
              # color = (255,0,0)
              color = (0, 0, 255)
              cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

       out.write(frame)
       frame_index += 1
    else:
        print('fine')
        break
time_end=time.time()
print('totally cost',time_end-time_start)
video_capture.release()
out.release()
cv2.destroyAllWindows()


