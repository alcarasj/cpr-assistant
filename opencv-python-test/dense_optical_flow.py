import cv2 as cv
import numpy as np


THRESHOLD = 1
LEARNING_RATE = 0.005
cap = cv.VideoCapture('Ken1BUV.mp4')
ret, frame1 = cap.read()
frame1 = cv.resize(frame1, (0, 0), fx=0.5, fy=0.5) 
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
weights = np.zeros_like(frame1[..., 0])
compressions = 0

while True:
    ret, frame2 = cap.read()
    orig = frame2
    frame2 = cv.resize(frame2, (0, 0), fx=0.5, fy=0.5)

    # Dense optical flow.
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev=prvs, next=next, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    #mag = np.array([[p if p >= THRESHOLD else 0 for p in row] for row in mag])
    #weights = np.array([[((1-LEARNING_RATE) * weight) + (LEARNING_RATE * mag[i][j]) for j, weight in enumerate(row)] for i, row in enumerate(weights)])
    #import pdb; pdb.set_trace()
    if np.sum(mag) > 185000:
        compressions += 1
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.namedWindow('Video', cv.WINDOW_NORMAL)
    cv.namedWindow('Flow', cv.WINDOW_NORMAL)
    cv.putText(orig, "(%i, %i)" % (np.sum(mag), np.mean(ang*180/np.pi/2)), (150,300), cv.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
    cv.putText(orig, "CMP: %i" % compressions, (150,200), cv.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
    cv.imshow('Video', orig)
    cv.imshow('Flow',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next

cap.release()
cv.destroyAllWindows()
