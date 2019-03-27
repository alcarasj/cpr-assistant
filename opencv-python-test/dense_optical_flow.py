import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Script execution configuration constants.
WEBCAM_MODE = False
READ_ONLY = False
SHOW_OUTPUT = True
SHOW_GT = False
INPUT_VIDEO = 'Jerico1BUV.mp4'
VIDEO = cv2.VideoCapture(0 if WEBCAM_MODE else INPUT_VIDEO)
FPS = int(VIDEO.get(cv2.CAP_PROP_FPS))
NUMBER_OF_FRAMES = float(VIDEO.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION = float(NUMBER_OF_FRAMES / FPS)
TEXT_START_POS_Y = 30
CSV_DIR = 'csv_results/%s.csv'
GT_DIR = 'Jerico1GT.mp4'
GT_VIDEO = cv2.VideoCapture(GT_DIR)
GT_CSV = './csv_raw/Jerico1GT.mp4.csv'

# Global constants for calculations (all times are in seconds).
MAX_ALLOWED_TIME_FOR_UPWARD_MOVEMENT = 0.7
BREATHS_MODE = True
MIN_FLOW_THRESHOLD = 0.33
SCALE = 0.33
LEARNING_RATE = 0.0075
MIN_RESULTANT = 5000
MIN_BREATHING_MOVEMENT = 10000
MIN_MOVEMENT_PCG = 20
AVERAGING_TIME = 0.2
AVERAGING_FRAMES = int(AVERAGING_TIME * FPS)


def plot_data(data):
    """ Plots data on a graph. """

    print("Plotting data...")
    frames = list(range(0, len(data)))
    time_in_seconds = [(i / FPS) for i in frames]
    movement = [coord[0] for coord in data]
    compressions = [coord[0] if coord[3] == "Compression" else None for coord in data]
    plt.plot(time_in_seconds, movement)
    plt.plot(time_in_seconds, compressions, "x")
    plt.legend(["Optical Flow", "Detected Compression (N=%i)" % len([c for c in compressions if c])])
    plt.ylabel('Resultant Vertical Displacement (Pixels)')
    plt.xlabel('Time (Seconds)')
    plt.axis([0, time_in_seconds[-1], -2000, 2000])
    plt.show()


def read_from_csv(csv_file):
    """ Reads data from a CSV file. """

    print("Reading from CSV file...")
    data = []

    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        data.append((float(row[0])/ 100, float(row[1])/ 100, row[2], row[3] if row[3] != 'None' else None))

    print("Reading from CSV file complete! %i co-ordinates read." % len(data))
    return data


def write_to_csv(data):
    """ Writes the data into a CSV file. """
    
    print("Writing to CSV file...")

    if os.path.exists(CSV_DIR % INPUT_VIDEO):
        os.remove(CSV_DIR % INPUT_VIDEO)

    try:
        os.stat('csv_results')
    except:
        os.mkdir('csv_results')
    
    csv_file = open(CSV_DIR % INPUT_VIDEO, mode='w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row_count = 0

    for coord in data:
        writer.writerow([coord[0], coord[1], coord[2], coord[3] if coord[3] else "None"])
        row_count += 1

    csv_file.close()
    print("Write to CSV file complete! %i/%i datapoints written." % (row_count, len(data))) 


def process_video():
    """ The core logic of this dissertation's solution - process the video to detect compressions. """

    # Initial values for use in calculations and output.
    (ret, current_frame_bgr) = VIDEO.read()
    print("ORIGINAL RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    print("DURATION IN SECONDS: %f" % DURATION)
    print("FPS: %f" % FPS)
    current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE) 
    hsv = np.zeros_like(current_frame_bgr)
    hsv[...,1] = 255
    current_frame_gray = cv2.cvtColor(current_frame_bgr,cv2.COLOR_BGR2GRAY)
    prev_frame_bgr = np.array([])
    weights = np.zeros_like(current_frame_bgr[..., 0])
    weights_mask = np.ones_like(current_frame_bgr[..., 0])
    weights_hsv = np.zeros_like(current_frame_bgr)
    weights_hsv[..., 1] = 0
    weights_hsv[..., 0] = 0
    prev_compression_time = None
    is_breathing = False
    elapsed_time = 0
    strong_downward_movement_detected = False
    upward_movement_detected_within_allowed_time = False
    strong_downward_movement_time = 0
    time_diff = 0
    data = []

    # Initial compression/CCR values.
    compressions = 0
    ccr = 0
    mean_ccr = 0
    ccr_data = np.array([])

    print("PROCESSED RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    total_pixels = len(current_frame_bgr[0]) * len(current_frame_bgr)
    print("TOTAL PIXELS: %i" % total_pixels)

    # Processing loop.
    while True:
        if prev_frame_bgr.any():
            (ret, current_frame_bgr) = VIDEO.read()
            current_frame_number = VIDEO.get(cv2.CAP_PROP_POS_FRAMES)

            if ret == False:
                break

            # Pre-processing to reduce noise.
            current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE)
            current_frame_bgr = cv2.blur(current_frame_bgr, (1, 1))

            # Dense optical flow.
            prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev=prev_frame_gray, next=current_frame_gray, flow=None, pyr_scale=0.5, levels=3, winsize=25, iterations=1, poly_n=5, poly_sigma=1.2, flags=0)
            magnitude, direction = cv2.cartToPolar(flow[...,0], flow[...,1])
            direction_in_deg = direction *  (180 / np.pi)

            # Apply thresholding to reduce noisy pixels. 
            np.place(magnitude, magnitude < MIN_FLOW_THRESHOLD, 0)
            
            # Isolate upward movements for detecting compressions. UP = 90deg.
            upward_movement_mask = np.zeros_like(magnitude)
            np.place(upward_movement_mask, np.logical_and(direction_in_deg > 45, direction_in_deg < 135), 1)
            upward_movement = cv2.bitwise_and(magnitude, magnitude, mask=upward_movement_mask.astype(np.int8))
            upward_movement = np.multiply(upward_movement, weights_mask)
            upward_sum = np.sum(upward_movement)

            # Isolate downward movements for detecting compressions. DOWN = 270deg.
            downward_movement_mask = np.zeros_like(magnitude)
            np.place(downward_movement_mask, np.logical_and(direction_in_deg > 225, direction_in_deg < 315), 1)
            downward_movement = cv2.bitwise_and(magnitude, magnitude, mask=downward_movement_mask.astype(np.int8))
            downward_movement = np.multiply(downward_movement, weights_mask)
            downward_sum = np.sum(downward_movement)

            # Isolate leftward movements for detecting breaths. LEFT = 180deg.
            leftward_movement_mask = np.zeros_like(magnitude)
            np.place(leftward_movement_mask, np.logical_and(direction_in_deg >= 170, direction_in_deg <= 190), 1)
            leftward_movement = cv2.bitwise_and(magnitude, magnitude, mask=leftward_movement_mask.astype(np.int8))
            leftward_movement = np.multiply(leftward_movement, weights_mask)
            leftward_sum = np.sum(leftward_movement)

            # Isolate rightward movements for detecting breaths. RIGHT = 0 or 360deg.
            rightward_movement_mask = np.zeros_like(magnitude)
            np.place(rightward_movement_mask, np.logical_or(direction_in_deg <= 10, direction_in_deg >= 350), 1)
            rightward_movement = cv2.bitwise_and(magnitude, magnitude, mask=rightward_movement_mask.astype(np.int8))
            rightward_movement = np.multiply(rightward_movement, weights_mask)
            rightward_sum = np.sum(rightward_movement)

            # Isolate lateral movements as a whole for visualisation.
            lateral_movement_mask = cv2.bitwise_or(leftward_movement_mask, rightward_movement_mask)
            lateral_movement = np.multiply(magnitude, lateral_movement_mask)
            lateral_sum = leftward_sum + rightward_sum

            # Outputs of calculations.
            total_movement_pcg = (np.sum(np.where(magnitude > 0)[0])) / total_pixels

            if is_breathing:
                state = None

            # Compute average motion over N previous frames, where N = AVERAGING_FRAMES.
            last_n_frames = data[-(AVERAGING_FRAMES):]          
            upward_avg = np.mean([frame[1] for frame in last_n_frames]) if data else 0
            downward_avg = np.mean([frame[2] for frame in last_n_frames]) if data else 0
            leftward_avg = np.mean([frame[5] for frame in last_n_frames]) if data else 0
            rightward_avg = np.mean([frame[6] for frame in last_n_frames]) if data else 0

            # Calculate weights for isolation of zone with high movement density.
            lateral_avg = leftward_avg + rightward_avg
            if lateral_avg > MIN_BREATHING_MOVEMENT or is_breathing:
                is_breathing = True
                state = "Breaths"

            if compressions > 0 and not is_breathing:
                old_weights = weights * (1 - LEARNING_RATE)
                temp = (magnitude * LEARNING_RATE)
                weights = np.add(old_weights, temp)
                weights = cv2.normalize(weights, weights, 0, 1, cv2.NORM_MINMAX)
                weights_mask = weights

            # Vertical resultant using averages for detecting compressions.
            vertical_resultant = upward_avg - downward_avg

            # Elapsed time from starting frame for calculating CCR.
            elapsed_time = float(current_frame_number / FPS)


            # COMPRESSION DETECTION.
            # 1. Detect a strong downward movement in the previous 200ms.
            # 2. If a strong downward movement is detected, try to detect a strong upward movement in the following 200ms.
            #    The strong upward movement must be detected within 500ms from the point in time when the strong downward movement was last detected.
            # 3. If these conditions are met, then a compression is detected by this algorithm.
            #    Otherwise, the downward movement is set back to False.

            if not strong_downward_movement_detected and vertical_resultant < -(MIN_RESULTANT) and total_movement_pcg > MIN_MOVEMENT_PCG:
                strong_downward_movement_detected = True
                strong_downward_movement_time = elapsed_time
            elif strong_downward_movement_detected and vertical_resultant > MIN_RESULTANT and total_movement_pcg > MIN_MOVEMENT_PCG and (elapsed_time - strong_downward_movement_time) <= MAX_ALLOWED_TIME_FOR_UPWARD_MOVEMENT:
                upward_movement_detected_within_allowed_time = True
            elif strong_downward_movement_detected and (elapsed_time - strong_downward_movement_time) > MAX_ALLOWED_TIME_FOR_UPWARD_MOVEMENT:
                strong_downward_movement_detected = False

            if strong_downward_movement_detected and upward_movement_detected_within_allowed_time:
                upward_movement_detected_within_allowed_time = False
                strong_downward_movement_detected = False
                compressions += 1
                state = "Compression"
                is_breathing = False

                # CCR is calculated as the time difference to complete two compressions, measured in BPM.
                if prev_compression_time:
                    time_diff = elapsed_time - prev_compression_time
                    ccr = 60 / time_diff
                    ccr_data = np.append(ccr_data, ccr)
                    mean_ccr = np.mean(ccr_data)

                prev_compression_time = elapsed_time


            print("%s[%i] (%ipc) VERT_R: %i %s" % ("----CCR: %f at %fs----" % (ccr, elapsed_time) if state else '', current_frame_number, total_movement_pcg, vertical_resultant, (", TDIFF: %f" % time_diff) if state == "Compression" else ""))
            data.append([vertical_resultant, int(upward_sum), int(downward_sum), state, int(total_movement_pcg), int(leftward_sum), int(rightward_sum)])


            # Show output windows for visualization.
            if SHOW_OUTPUT:
                hsv[..., 0] = cv2.normalize(direction_in_deg, None, 0, 179, cv2.NORM_MINMAX)
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(upward_movement, None, 0, 255, cv2.NORM_MINMAX)
                flow_up_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(downward_movement, None, 0, 255, cv2.NORM_MINMAX)      
                flow_down_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(lateral_movement, None, 0, 255, cv2.NORM_MINMAX)      
                flow_lateral_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                weights_hsv[..., 2] = cv2.normalize(weights_mask, None, 0, 255, cv2.NORM_MINMAX)
                weights_bgr = cv2.cvtColor(weights_hsv, cv2.COLOR_HSV2BGR)


                cv2.putText(flow_bgr, "Time: %f" % elapsed_time, (25, TEXT_START_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "CCR: %fbpm" % ccr, (25, TEXT_START_POS_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "S: %s" % (state), (25, TEXT_START_POS_Y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "Nc: %i" % compressions, (25, TEXT_START_POS_Y + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "TDIFF: %f" % time_diff, (25, TEXT_START_POS_Y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (All)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (Upward)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (Downward)', cv2.WINDOW_NORMAL)
                cv2.imshow('Weighted Mask', cv2.WINDOW_NORMAL)
                cv2.imshow('Video', current_frame_bgr)
                cv2.imshow('Flow (All)', flow_bgr)
                cv2.imshow('Flow (Upward)', flow_up_bgr)
                cv2.imshow('Flow (Downward)', flow_down_bgr)
                cv2.imshow('Flow (Lateral)', flow_lateral_bgr)
                cv2.imshow('Weighted Mask', weights_bgr)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

        prev_frame_bgr = current_frame_bgr

    VIDEO.release()
    cv2.destroyAllWindows()
    return data


def main():
    if not READ_ONLY:
        data = process_video()
        write_to_csv(data)
    else:
        try:
            existing_csv = open(CSV_DIR % INPUT_VIDEO)
            data = read_from_csv(existing_csv)
        except FileNotFoundError:
            data = process_video()
            write_to_csv(data)

    plot_data(data)

if __name__ == '__main__':
    main()
