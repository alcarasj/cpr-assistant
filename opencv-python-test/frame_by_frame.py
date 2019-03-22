import cv2
import imutils
import numpy as np


cap = cv2.VideoCapture('Ken1BUV.mp4')
TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)


frames = []

while True:
	(ret, frame) = cap.read()
	frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
	if frame_number <= TOTAL_FRAMES and frame is not None:
		frame = imutils.rotate_bound(frame, 180)
		cv2.namedWindow('swag', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('swag', 800,800)
		cv2.putText(frame, "%i" % frame_number, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), thickness=2)
		cv2.imshow('swag', frame)
		if cv2.waitKey(0) & 0xFF == ord('n'):
			pass
		else:
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
