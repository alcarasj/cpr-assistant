import cv2
import numpy as np

GRD_TRUTH_VIDEO = cv2.VideoCapture("./Ken1FCSV.mp4")
GRD_TRUTH_START_FRAME = 69
GRD_TRUTH_FPS = int(GRD_TRUTH_VIDEO.get(cv2.CAP_PROP_FPS))
BUV_VIDEO = cv2.VideoCapture("./Ken1BUV.mp4")
BUV_START_FRAME = 147
BUV_FPS = int(BUV_VIDEO.get(cv2.CAP_PROP_FPS))
GT_START_DURATION = GRD_TRUTH_START_FRAME / GRD_TRUTH_FPS
BUV_START_DURATION = BUV_START_FRAME / BUV_FPS

print("")
print("")

while True:
	ret1, frame1 = BUV_VIDEO.read()
	frame1_num = int(BUV_VIDEO.get(cv2.CAP_PROP_POS_FRAMES))
	cv2.namedWindow('Ground Truth', cv2.WINDOW_NORMAL)
	cv2.namedWindow('Bottom-Up View', cv2.WINDOW_NORMAL)
	if ret1 == True and frame1_num % BUV_FPS == 0 and frame1_num >= BUV_START_FRAME:
		while True:
			ret2, frame2 = GRD_TRUTH_VIDEO.read()
			frame2_num = int(GRD_TRUTH_VIDEO.get(cv2.CAP_PROP_POS_FRAMES))
			if ret2 == True and frame2_num % GRD_TRUTH_FPS == 0 and frame2_num >= GRD_TRUTH_START_FRAME:
				buv_dur = (BUV_VIDEO.get(cv2.CAP_PROP_POS_MSEC) / 1000) - BUV_START_DURATION
				print("%f" % buv_dur)
				gt_dur = (GRD_TRUTH_VIDEO.get(cv2.CAP_PROP_POS_MSEC) / 1000) - GT_START_DURATION
				print("%f" % buv_dur)
				cv2.imshow('Ground Truth', frame2)
				cv2.imshow('Bottom-Up View', frame1)
				cv2.waitKey(0)
				break


	cv2.waitKey(1)

GRD_TRUTH_VIDEO.release()
BUV_VIDEO.release()
cv2.destroyAllWindows()




