import cv2
import numpy as np

GRD_TRUTH_VIDEO = cv2.VideoCapture("./Ken1FCSV.mp4")
GRD_TRUTH_START_FRAME = 69
GRD_TRUTH_FPS = int(GRD_TRUTH_VIDEO.get(cv2.CAP_PROP_FPS))
BUV_VIDEO = cv2.VideoCapture("./Ken1BUV.mp4")
BUV_START_FRAME = 146
BUV_FPS = int(BUV_VIDEO.get(cv2.CAP_PROP_FPS))
ALIGNMENT_FACTOR_IS_BUV = GRD_TRUTH_FPS > BUV_FPS

while True:
	ret1, grd_truth_frame  = GRD_TRUTH_VIDEO.read()
	grd_truth_frame_num = int(GRD_TRUTH_VIDEO.get(cv2.CAP_PROP_POS_FRAMES))
	cv2.namedWindow('Ground Truth', cv2.WINDOW_NORMAL)
	cv2.namedWindow('Bottom-Up View', cv2.WINDOW_NORMAL)

	if ret1:
		if grd_truth_frame_num % GRD_TRUTH_FPS == 0 and grd_truth_frame_num >= GRD_TRUTH_START_FRAME:
			cv2.imshow('Ground Truth', grd_truth_frame)
			while True:
				ret2, buv_frame = BUV_VIDEO.read()
				buv_frame_num = int(BUV_VIDEO.get(cv2.CAP_PROP_POS_FRAMES))
				if buv_frame_num % BUV_FPS == 0 and buv_frame_num >= BUV_START_FRAME:
					cv2.imshow('Bottom-Up View', buv_frame)
					cv2.waitKey(1)
					break


	cv2.waitKey(1)

GRD_TRUTH_VIDEO.release()
BUV_VIDEO.release()
cv2.destroyAllWindows()




