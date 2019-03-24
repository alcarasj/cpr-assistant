import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

WEBCAM_MODE = False
READ_ONLY = False
INPUT_VIDEO = './Ken1BUV.mp4'
MIN_TIME_BETWEEN_MAXIMUMS = 0.2
STRIDE = 8
MIN_FLOW_THRESHOLD = 0.75
WEIGHTING_THRESHOLD = 0.2
START_FRAME = 146
SCALE = 0.33
LEARNING_RATE = 0.008
VIDEO = cv2.VideoCapture(0 if WEBCAM_MODE else INPUT_VIDEO)
FPS = VIDEO.get(cv2.CAP_PROP_FPS)
NUMBER_OF_FRAMES = int(VIDEO.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION = float(NUMBER_OF_FRAMES - START_FRAME) / float(FPS)
TEXT_START_POS_Y = 30
CSV_DIR = 'csv_results/%s.csv'


def plot_data(data):
    """ Plots the raw data (x, y) data into a graph. """

    print("Plotting data...")
    frames = list(range(1, len(data) + 1))
    movement = [coord[1] for coord in data]
    plt.plot(frames, movement)
    plt.ylabel('Movement')
    plt.xlabel('Frame')
    plt.axis([0, len(data) * 1.5, -5000, 5000])
    plt.show()


def read_from_csv(csv_file):
    """ Reads data from a CSV file. """

    print("Reading from CSV file...")
    coords = []

    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        coords.append((float(row[0])/ 100, int(row[1])/ 100))

    print("Reading from CSV file complete! %i co-ordinates read." % len(coords))
    return coords


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
        writer.writerow([coord[0], coord[1], coord[2]])

    csv_file.close()
    print("Write to CSV file complete! %i/%i datapoints written." % (row_count, len(data))) 


def process_video():

    # Initial values.
    (ret, current_frame_bgr) = VIDEO.read()
    print("ORIGINAL RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE) 
    hsv = np.zeros_like(current_frame_bgr)
    hsv[...,1] = 255
    compressions = 0
    current_frame_gray = cv2.cvtColor(current_frame_bgr,cv2.COLOR_BGR2GRAY)
    prev_frame_bgr = np.array([])
    weights = np.zeros_like(current_frame_bgr[..., 0])
    weights_mask = np.ones_like(current_frame_bgr[..., 0])
    weights_hsv = np.zeros_like(current_frame_bgr)
    weights_hsv[..., 1] = 0
    weights_hsv[..., 0] = 0
    minimum = 0
    prev_compression_time = None
    prev_resultant = 0
    elapsed_time = 0
    ccr = 0
    mean_ccr = 0
    ccr_data = np.array([])
    print("PROCESSED RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    total_pixels = len(current_frame_bgr[0]) * len(current_frame_bgr)
    print("TOTAL PIXELS: %i" % total_pixels)

    data = []

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
            direction_in_deg = direction * 180 / np.pi / 2

            # Apply thresholding.
            np.place(magnitude, magnitude <= MIN_FLOW_THRESHOLD, 0)
            downward_movement_mask = np.zeros_like(magnitude)
            upward_movement_mask = np.zeros_like(magnitude)

            # Isolate downward movements.
            np.place(downward_movement_mask, np.logical_and(direction_in_deg > 100, direction_in_deg < 260), 1)
            downward_movement = cv2.bitwise_and(magnitude, magnitude, mask=downward_movement_mask.astype(np.int8))
            downward_movement = cv2.bitwise_and(downward_movement, downward_movement, mask=weights_mask)
            downward_sum = np.sum(downward_movement)

            # Isolate upward movements.
            np.place(upward_movement_mask, np.logical_or(direction_in_deg > 280, direction_in_deg < 70), 1)
            upward_movement = cv2.bitwise_and(magnitude, magnitude, mask=upward_movement_mask.astype(np.int8))
            upward_movement = cv2.bitwise_and(upward_movement, upward_movement, mask=weights_mask)
            upward_sum = np.sum(upward_movement)

            total_movement_pcg = (np.sum(np.where(downward_movement > 0)[0]) + np.sum(np.where(upward_movement > 0)[0])) / total_pixels
            vertical_resultant = upward_sum - downward_sum
            five_frame_avg = sum([val[0] for val in data[-5:]]) / 5

            data.append((vertical_resultant, int(five_frame_avg), int(total_movement_pcg)))

            if current_frame_number > START_FRAME or WEBCAM_MODE:
                # Calculate weights for isolation of zone with high movement density.

                old_weights = weights * (1 - LEARNING_RATE)
                temp = (magnitude * LEARNING_RATE)
                weights = np.add(old_weights, temp)
                weights = cv2.normalize(weights, None, 0, 1, cv2.NORM_MINMAX)
                np.place(weights_mask, weights >= WEIGHTING_THRESHOLD, 1)
                np.place(weights_mask, weights < WEIGHTING_THRESHOLD, 0)

                # Elapsed time for calculating CCR.
                elapsed_time = float(current_frame_number - START_FRAME) / float(FPS)

                # Compression detection.
                if vertical_resultant > prev_resultant and minimum == prev_resultant:
                    compressions += 1
                    minimum = 0

                    # CCR is calculated as the time difference to complete two compressions, measured in BPM.
                    if prev_compression_time:
                        time_diff = elapsed_time - prev_compression_time
                        ccr = 60 / (elapsed_time - prev_compression_time)

                        if time_diff > MIN_TIME_BETWEEN_MAXIMUMS:
                            ccr_data = np.append(ccr_data, ccr)
                            mean_ccr = np.mean(ccr_data)

                    prev_compression_time = elapsed_time
                if vertical_resultant < prev_resultant:
                    minimum = prev_resultant

            # Outputs.
            hsv[..., 0] = cv2.normalize(direction_in_deg, None, 0, 179, cv2.NORM_MINMAX)
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            hsv[..., 2] = cv2.normalize(upward_movement, None, 0, 255, cv2.NORM_MINMAX)
            flow_up_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            hsv[..., 2] = cv2.normalize(downward_movement, None, 0, 255, cv2.NORM_MINMAX)      
            flow_down_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            weights_hsv[..., 2] = cv2.normalize(weights_mask, None, 0, 255, cv2.NORM_MINMAX)
            weights_bgr = cv2.cvtColor(weights_hsv, cv2.COLOR_HSV2BGR)

            print("[%i] = %i (%ipc) 5FA=%i" % (current_frame_number, vertical_resultant, total_movement_pcg, five_frame_avg))

            cv2.putText(flow_bgr, "Time: %f" % elapsed_time, (25, TEXT_START_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
            cv2.putText(flow_bgr, "CCR: %fbpm" % ccr, (25, TEXT_START_POS_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
            cv2.putText(flow_bgr, "AVGCCR: %fbpm" % mean_ccr, (25, TEXT_START_POS_Y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
            cv2.putText(flow_bgr, "VECTOR: %f" % vertical_resultant, (25, TEXT_START_POS_Y + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
            cv2.putText(flow_bgr, "Nc: %i" % compressions, (25, TEXT_START_POS_Y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Flow (All)', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Flow (Upward)', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Flow (Downward)', cv2.WINDOW_NORMAL)
            cv2.imshow('Weighted Mask', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', current_frame_bgr)
            cv2.imshow('Flow (All)', flow_bgr)
            cv2.imshow('Flow (Upward)', flow_up_bgr)
            cv2.imshow('Flow (Downward)', flow_down_bgr)
            cv2.imshow('Weighted Mask', weights_bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            prev_resultant = vertical_resultant

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
