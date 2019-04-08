import cv2
import imutils
import numpy as np


cap = cv2.VideoCapture('./videos/GT/L3_GT.mp4')
TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
FPS = float(cap.get(cv2.CAP_PROP_FPS))
START_FRAME = 0
START_DURATION = START_FRAME / FPS


print("FPS: %i" % FPS)

while True:
	(ret, frame) = cap.read()
	frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
	duration_from_start = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) - START_DURATION
	if frame_number >= START_FRAME and frame_number <= TOTAL_FRAMES and frame is not None:
		frame = imutils.rotate_bound(frame, 90)
		cv2.namedWindow('swag', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('swag', 800,800)
		cv2.putText(frame, "%i" % (frame_number - START_FRAME), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=5)
		cv2.putText(frame, "%f" % duration_from_start, (200, 275), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=5)
		cv2.imshow('swag', frame)
		if cv2.waitKey(0) & 0xFF == ord('n'):
			pass
		else:
			break

cap.release()
cv2.destroyAllWindows()
