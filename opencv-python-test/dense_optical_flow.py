import cv2
import numpy as np

VIDEO_DIR = 'Ken1BUV.mp4'
MIN_TIME_BETWEEN_MAXIMUMS = 0.2
MIN_FLOW_THRESHOLD = 1.25
START_FRAME = 146
SCALE = 0.33
LEARNING_RATE = 0.005
VIDEO = cv2.VideoCapture(VIDEO_DIR)
FPS = VIDEO.get(cv2.CAP_PROP_FPS)
NUMBER_OF_FRAMES = int(VIDEO.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION = float(NUMBER_OF_FRAMES - START_FRAME) / float(FPS)
TEXT_START_POS_Y = 30

def main():

    # Initial values.
    (ret, current_frame_bgr) = VIDEO.read()
    print("ORIGINAL RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE) 
    hsv = np.zeros_like(current_frame_bgr)
    hsv[...,1] = 255
    compressions = 0
    current_frame_gray = cv2.cvtColor(current_frame_bgr,cv2.COLOR_BGR2GRAY)
    prev_frame_bgr = np.array([])
    maximum = 0
    prev_compression_time = None
    elapsed_time = 0
    ccr = 0
    mean_ccr = 0
    prev_sum_magnitude = 0
    ccr_data = np.array([])
    print("PROCESSED RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))

    # Processing loop.
    while True:
        if prev_frame_bgr.any():
            (ret, current_frame_bgr) = VIDEO.read()
            current_frame_number = int(VIDEO.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame_bgr is None:
                break

            # Pre-processing to eliminate noise.
            current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE)
            current_frame_bgr = cv2.blur(current_frame_bgr, (1, 1))

            # Dense optical flow.
            prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev=prev_frame_gray, next=current_frame_gray, flow=None, pyr_scale=0.5, levels=3, winsize=25, iterations=1, poly_n=5, poly_sigma=1.2, flags=0)
            magnitude, direction = cv2.cartToPolar(flow[...,0], flow[...,1])

            # Apply thresholding.
            np.place(magnitude, magnitude <= MIN_FLOW_THRESHOLD, 0)
            direction_in_deg = direction * 180 / np.pi / 2
            mean_direction = np.mean(direction_in_deg)
            current_sum_magnitude = np.sum(magnitude)

            if current_frame_number > START_FRAME:

                # Elapsed time for calculating CCR.
                elapsed_time = float(current_frame_number - START_FRAME) / float(FPS)

                # Compression detection.
                if current_sum_magnitude < prev_sum_magnitude and prev_sum_magnitude == maximum and mean_direction < 80:
                    compressions += 1
                    maximum = 0

                    # CCR is calculated as the time difference to complete two compressions, measured in BPM.
                    if prev_compression_time:
                        time_diff = elapsed_time - prev_compression_time
                        ccr = 60 / (elapsed_time - prev_compression_time)

                        if time_diff > MIN_TIME_BETWEEN_MAXIMUMS:
                            ccr_data = np.append(ccr_data, ccr)
                            mean_ccr = np.mean(ccr_data)

                    prev_compression_time = elapsed_time

                # Deduce if current magnitude is a maximum.
                if current_sum_magnitude > prev_sum_magnitude:
                    maximum = current_sum_magnitude

            # Outputs.
            hsv[...,0] = direction_in_deg
            hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            print("[%i] - (%i, %i)" % (current_frame_number, current_sum_magnitude, mean_direction))
            cv2.putText(flow_bgr, "Time: %f" % elapsed_time, (25, TEXT_START_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.putText(flow_bgr, "CCR: %fbpm" % ccr, (25, TEXT_START_POS_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.putText(flow_bgr, "AVGCCR: %fbpm" % mean_ccr, (25, TEXT_START_POS_Y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.putText(flow_bgr, "SUMMAG: %i" % current_sum_magnitude, (25, TEXT_START_POS_Y + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.putText(flow_bgr, "AVGDIR: %i" % mean_direction, (25, TEXT_START_POS_Y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.putText(flow_bgr, "Nc: %i" % compressions, (25, TEXT_START_POS_Y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, 255, thickness=1)
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Flow', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', current_frame_bgr)
            cv2.imshow('Flow', flow_bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            prev_sum_magnitude = current_sum_magnitude

        prev_frame_bgr = current_frame_bgr

    VIDEO.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
