import cv2 as cv
import numpy as np


THRESHOLD = 1
SCALE = 0.5
LEARNING_RATE = 0.005
VIDEO_DIR = 'Ken1BUV.mp4'

def main():
    video = cv.VideoCapture(VIDEO_DIR)
    (ret, next_frame_bgr) = video.read()
    next_frame_bgr = cv.resize(next_frame_bgr, (0, 0), fx=SCALE, fy=SCALE) 
    hsv = np.zeros_like(next_frame_bgr)
    hsv[...,1] = 255
    compressions = 0
    next_frame_gray = cv.cvtColor(next_frame_bgr,cv.COLOR_BGR2GRAY)
    prev_frame_bgr = np.array([])

    while True:
        if prev_frame_bgr.any():
            (ret, next_frame_bgr) = video.read()

            # Pre-processing to eliminate noise.
            next_frame_bgr = cv.resize(next_frame_bgr, (0, 0), fx=SCALE, fy=SCALE)
            next_frame_bgr = cv.GaussianBlur(next_frame_bgr, (5, 5), 5)

            # Dense optical flow.
            prev_frame_gray = cv.cvtColor(prev_frame_bgr, cv.COLOR_BGR2GRAY)
            next_frame_gray = cv.cvtColor(next_frame_bgr, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev=prev_frame_gray, next=next_frame_gray, flow=None, pyr_scale=0.5, levels=3, winsize=25, iterations=1, poly_n=5, poly_sigma=1.2, flags=0)
            magnitude, direction = cv.cartToPolar(flow[...,0], flow[...,1])
            np.place(magnitude, magnitude <= THRESHOLD, 0)
            direction_in_deg = direction * 180 / np.pi / 2
            mean_direction = np.mean(direction_in_deg)

            # Compression detection.
            if np.sum(magnitude) > 185000 and mean_direction < 90:
                compressions += 1

            # Output windows.
            original = next_frame_bgr
            hsv[...,0] = direction_in_deg
            hsv[...,2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
            flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.namedWindow('Video', cv.WINDOW_NORMAL)
            cv.namedWindow('Flow', cv.WINDOW_NORMAL)
            cv.imshow('Video', original)
            cv.imshow('Flow', flow_bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        prev_frame_bgr = next_frame_bgr

    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
