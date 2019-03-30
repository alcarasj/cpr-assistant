import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser


parser = ArgumentParser(description='Compressions/breaths detector for CPR using dense optical flow.')
parser.add_argument('-d', '--dataset', dest='dataset', help='The name of the dataset e.g. Jerico1.', type=str, required=True)
parser.add_argument('-r', '--recalculate', dest='recalculate', help='Boolean flag to recalculate if there is any existing CSV (also overwrites existing CSV).', action='store_true')
parser.add_argument('-o', '--output', dest='output', help='Boolean flag to show output windows.',  action='store_true')
parser.add_argument('-w', '--weights-file', dest='weights_file', help='Relative path to an .npy weights file to preload weights before processing.', type=str, required=False)
parser.add_argument('-s', '--save-weights', dest='save_weights', help='Save the weights after processing.', action='store_true')
args = parser.parse_args()


""""
HYPERPARAMETERS (all times are in seconds).

BREATHING_MODE - Detects breathing if True.
LEARNING_RATE - Learning rate for a weighted mask for isolation of user in frame (experimental ideal is 0.0065).
LOOKBACK_TIME - The time for which the algorithm calculates derivatives of movement (velocity and acceleration).
MINIMUM_ACCELERATION - The minimum acceleration (in pixels/s/s) required for detecting a compression.
MOVING_AVG_PERIOD - The moving average period length.
MAX_TIME_FOR_UPWARD_ACCELERATION - The max allowed time for an upward acceleration to be detected as a compression from when a strong downward acceleration was detected.
MIN_MOVEMENT_PCG - The minimum percentage of pixels moved to be considered valid for compression detection.
MIN_FLOW_THRESHOLD - The minimum flow magnitude to be considered not to be noise.
SCALE - The scale factor for the frame before processing.

"""
BREATHING_MODE = True  
LEARNING_RATE = 0.005
LOOKBACK_TIME = 0.3
MINIMUM_ACCELERATION = 1000
MOVING_AVG_PERIOD = 0.3
MAX_TIME_FOR_UPWARD_ACCELERATION = 0.75
MIN_MOVEMENT_PCG = 15
MIN_FLOW_THRESHOLD = 0.2
SCALE = 0.2
MIN_BREATHING_MOVEMENT = 5000

# Arguments. 
DATASET = args.dataset
RECALCULATE = args.recalculate
SHOW_OUTPUT = args.output
WEIGHTS_FILE = args.weights_file
SAVE_WEIGHTS = args.save_weights

# Files and miscellaneous constants.
INPUT_VIDEO = DATASET + 'BUV.mp4'
WEIGHTS_DIR = "./weights/%s.npy" % DATASET 
VIDEO = cv2.VideoCapture(INPUT_VIDEO)
FPS = int(VIDEO.get(cv2.CAP_PROP_FPS))
NUMBER_OF_FRAMES = int(VIDEO.get(cv2.CAP_PROP_FRAME_COUNT))
DURATION = float(NUMBER_OF_FRAMES / FPS)
TEXT_START_POS_Y = 30
CSV_DIR = 'csv_results/%s.csv' % INPUT_VIDEO
GT_DIR = DATASET + 'GT.mp4'
GT_VIDEO = cv2.VideoCapture(GT_DIR)
GT_CSV_DIR = './csv_gt/' + DATASET + 'GT.mp4.csv'
AVERAGING_FRAMES = int(MOVING_AVG_PERIOD * FPS)
LOOKBACK_FRAMES = int(LOOKBACK_TIME * FPS)


def save_weights(weights):
    """ Saves the weights after the processing of video. """

    if os.path.exists(WEIGHTS_DIR):
        os.remove(WEIGHTS_DIR)

    try:
        os.stat('weights')
    except:
        os.mkdir('weights')

    np.save(WEIGHTS_DIR, weights)
    print("Weights saved successfully!") 


def load_weights(weights_file):
    """ Preloads existing weights for use in the processing of video. """

    return np.load(weights_file)


def plot_data(data, ground_truth=None):
    """ Plots data on a graph. """
    print("Plotting data...")
    window, chart = plt.subplots(2, sharex=True)
    window.suptitle("DATASET: " + DATASET)

    # Chart for solution.
    frames = list(range(0, len(data)))
    time_in_seconds = [(i / FPS) for i in frames]
    movement_delta = [coord[4] if not coord[3] or coord[3] == "Compression" else None for coord in data]
    breathing = [coord[4] if coord[3] == "Breathing" or coord[3] == "Compression" else None for coord in data]
    compressions = [coord[4] if coord[3] == "Compression" else None for coord in data]
    chart[0].set_title('Solution')
    chart[0].plot(time_in_seconds, movement_delta, "y-", color="orange")
    chart[0].plot(time_in_seconds, breathing, "y:", color="orange")
    chart[0].plot(time_in_seconds, compressions, "r*")
    chart[0].legend(["Optical Flow", "Optical Flow (Breathing)", "Detected Compression (N=%i)" % len([c for c in compressions if c])])
    chart[0].set_ylabel('Vertical Acceleration (pixels/s/s)')
    chart[0].axis([0, time_in_seconds[-1], np.min([c for c in movement_delta if c]) * 1.2, np.max([c for c in movement_delta if c]) * 1.2])

    # Chart for ground truth.
    frames = list(range(0, len(ground_truth)))
    time_in_seconds = [(i / FPS) for i in frames]
    movement = [coord[1] if not coord[3] or coord[3] == "Compression" else None for coord in ground_truth]
    breathing = [coord[1] if coord[3] == "Breathing" or coord[3] == "Compression" else None for coord in ground_truth]
    compressions = [coord[1] if coord[3] == "Compression" else None for coord in ground_truth]
    chart[1].set_title('Ground Truth')
    chart[1].plot(time_in_seconds, movement, "b-")
    chart[1].plot(time_in_seconds, breathing, "b:")
    chart[1].plot(time_in_seconds, compressions, "m^")
    chart[1].legend(["Ground Truth", "Ground Truth (Breathing)", "True Compression (N=%i)" % len([c for c in compressions if c])])
    chart[1].set_ylabel('Y Co-ordinates')
    chart[1].set_xlabel('Time (Seconds)')
    chart[1].axis([0, time_in_seconds[-1], np.min([c for c in movement if c]) * 1.2, np.max([c for c in movement if c]) * 1.2])
    
    plt.show()


def read_from_csv(csv_file):
    """ Reads data from a CSV file. """

    print("Reading from CSV file...")
    data = []

    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        try:
            data.append((float(row[0]) if row[0] != '-' else None, float(row[1]) if row[1] != '-' else None, row[2], row[3] if row[3] != 'None' else None, float(row[4])))
        except IndexError:
            data.append((float(row[0]) if row[0] != '-' else None, float(row[1]) if row[1] != '-' else None, row[2], row[3] if row[3] != 'None' else None))

    print("Reading from CSV file complete! %i co-ordinates read." % len(data))
    return data


def write_to_csv(data):
    """ Writes the data into a CSV file. """
    
    print("Writing to CSV file...")

    if os.path.exists(CSV_DIR):
        os.remove(CSV_DIR)

    try:
        os.stat('csv_results')
    except:
        os.mkdir('csv_results')
    
    csv_file = open(CSV_DIR, mode='w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row_count = 0

    for coord in data:
        writer.writerow([coord[0], coord[1], coord[2], coord[3] if coord[3] else "None", coord[8]])
        row_count += 1

    csv_file.close()
    print("Write to CSV file complete! %i/%i datapoints written." % (row_count, len(data))) 


def process_video(ground_truth, preloaded_weights=None):
    """ The core logic of this dissertation's solution - process the video to detect compressions. """

    # Initial values for use in calculations and output.
    GT_VIDEO.read()
    (ret, current_frame_bgr) = VIDEO.read()
    print("ORIGINAL RESOLUTION: %ix%i" % (len(current_frame_bgr[0]), len(current_frame_bgr)))
    print("DURATION IN SECONDS: %f" % DURATION)
    print("FPS: %f" % FPS)
    current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE) 
    hsv = np.zeros_like(current_frame_bgr)
    hsv[...,1] = 255
    current_frame_gray = cv2.cvtColor(current_frame_bgr,cv2.COLOR_BGR2GRAY)
    prev_frame_bgr = np.array([])
    prev_compression_time = None
    is_breathing = False
    state = None
    elapsed_time = 0
    strong_downward_acceleration_detected = False
    upward_movement_detected_within_allowed_time = False
    strong_downward_acceleration_time = 0
    time_diff = 0
    data = []

    # Preload weights or start weights from scratch.
    weights_hsv = np.zeros_like(current_frame_bgr)
    weights_hsv[..., 1] = 0
    weights_hsv[..., 0] = 0
    if preloaded_weights is not None:
        weights = preloaded_weights * 0.1
    else: 
        weights = np.zeros_like(current_frame_bgr[..., 0]) + 0.5

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
            (ret1, current_frame_bgr) = VIDEO.read()
            (ret2, gt_frame) = GT_VIDEO.read()
            current_frame_number = VIDEO.get(cv2.CAP_PROP_POS_FRAMES)

            if ret1 == False or ret2 == False:
                break

            # Pre-processing to reduce noise.
            current_frame_bgr = cv2.resize(current_frame_bgr, (0, 0), fx=SCALE, fy=SCALE)
            current_frame_bgr = cv2.blur(current_frame_bgr, (1, 1))

            # Dense optical flow.
            prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev=prev_frame_gray,
                                                next=current_frame_gray, 
                                                flow=None, 
                                                pyr_scale=0.5, 
                                                levels=3, 
                                                winsize=8, 
                                                iterations=1, 
                                                poly_n=3,
                                                poly_sigma=1.1, 
                                                flags=0)
            magnitude, direction = cv2.cartToPolar(flow[...,0], flow[...,1])
            direction_in_deg = direction *  (180 / np.pi)

            # Apply thresholding to reduce noisy pixels. 
            np.place(magnitude, magnitude < MIN_FLOW_THRESHOLD, 0)
            
            # Isolate upward movements for detecting compressions. UP = 90deg.
            upward_movement_mask = np.zeros_like(magnitude)
            np.place(upward_movement_mask, np.logical_and(direction_in_deg > 45, direction_in_deg < 135), 1)
            upward_movement = cv2.bitwise_and(magnitude, magnitude, mask=upward_movement_mask.astype(np.int8))
            upward_movement_weighted = np.multiply(upward_movement, weights)
            upward_sum = np.sum(upward_movement_weighted)

            # Isolate downward movements for detecting compressions. DOWN = 270deg.
            downward_movement_mask = np.zeros_like(magnitude)
            np.place(downward_movement_mask, np.logical_and(direction_in_deg > 225, direction_in_deg < 315), 1)
            downward_movement = cv2.bitwise_and(magnitude, magnitude, mask=downward_movement_mask.astype(np.int8))
            downward_movement_weighted = np.multiply(downward_movement, weights)
            downward_sum = np.sum(downward_movement_weighted)

            # Isolate leftward movements for detecting breaths. LEFT = 180deg.
            leftward_movement_mask = np.zeros_like(magnitude)
            np.place(leftward_movement_mask, np.logical_and(direction_in_deg >= 160, direction_in_deg <= 200), 1)
            leftward_movement = cv2.bitwise_and(magnitude, magnitude, mask=leftward_movement_mask.astype(np.int8))
            leftward_movement = np.multiply(leftward_movement, weights)
            leftward_sum = np.sum(leftward_movement)

            # Isolate rightward movements for detecting breaths. RIGHT = 0 or 360deg.
            rightward_movement_mask = np.zeros_like(magnitude)
            np.place(rightward_movement_mask, np.logical_or(direction_in_deg <= 20, direction_in_deg >= 340), 1)
            rightward_movement = cv2.bitwise_and(magnitude, magnitude, mask=rightward_movement_mask.astype(np.int8))
            rightward_movement = np.multiply(rightward_movement, weights)
            rightward_sum = np.sum(rightward_movement)

            # Isolate lateral movements as a whole for visualisation.
            lateral_movement = rightward_movement + leftward_movement
            lateral_sum = leftward_sum + rightward_sum

            # Outputs of calculations.
            total_movement_pcg = (np.sum(np.where(np.multiply(magnitude, weights) > 0)[0])) / total_pixels

            if is_breathing:
                state = None

            # Compute average motion over N previous frames, where N = AVERAGING_FRAMES.
            last_n_frames = data[-(AVERAGING_FRAMES):]          
            upward_avg = np.mean([frame[1] for frame in last_n_frames]) if data else 0
            downward_avg = np.mean([frame[2] for frame in last_n_frames]) if data else 0
            leftward_avg = np.mean([frame[5] for frame in last_n_frames]) if data else 0
            rightward_avg = np.mean([frame[6] for frame in last_n_frames]) if data else 0

            # Vertical resultant for detecting compressions.
            vertical_displacement = upward_avg - downward_avg

            # Get vertical delta for detecting compressions.
            if last_n_frames and len(last_n_frames) >= LOOKBACK_FRAMES:
                vertical_velocity = vertical_displacement - data[-(LOOKBACK_FRAMES)][0]
                vertical_acceleration = vertical_velocity - data[-(LOOKBACK_FRAMES)][7]
            else: 
                vertical_velocity = 0
                vertical_acceleration = 0

            # Elapsed time from starting frame for calculating CCR.
            elapsed_time = float(current_frame_number / FPS)

            # If there is large lateral movement observed, it is breathing.
            lateral_avg = leftward_avg + rightward_avg
            if BREATHING_MODE and (lateral_avg > MIN_BREATHING_MOVEMENT or is_breathing) and compressions > 10:
                is_breathing = True
                state = "Breathing"

            # COMPRESSION DETECTION.
            # 1. Detect a strong downward acceleration in the previous X seconds defined by MOVING_AVG_PERIOD.
            # 2. If a strong downward acceleration is detected, try to detect a strong upward acceleration in the following Y seconds defined by MAX_TIME_FOR_UPWARD_ACCELERATION.
            # 3. If these conditions are met, then a compression is detected by this algorithm.
            #    Otherwise, the downward movement is set back to False.


            acc_time_diff = elapsed_time - strong_downward_acceleration_time
            if not strong_downward_acceleration_detected and vertical_acceleration < -(MINIMUM_ACCELERATION * 0.25) and total_movement_pcg > MIN_MOVEMENT_PCG:
                strong_downward_acceleration_detected = True
                strong_downward_acceleration_time = elapsed_time
            elif strong_downward_acceleration_detected and vertical_acceleration > (MINIMUM_ACCELERATION) and total_movement_pcg > MIN_MOVEMENT_PCG and acc_time_diff <= MAX_TIME_FOR_UPWARD_ACCELERATION:
                upward_movement_detected_within_allowed_time = True
            elif strong_downward_acceleration_detected and (elapsed_time - strong_downward_acceleration_time) > MAX_TIME_FOR_UPWARD_ACCELERATION:
                strong_downward_acceleration_detected = False

            if strong_downward_acceleration_detected and upward_movement_detected_within_allowed_time:
                upward_movement_detected_within_allowed_time = False
                strong_downward_acceleration_detected = False
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

            print("[%i] (%ipc) VRT_DELTA: %i, LAT_R: %i, S: %s" % (current_frame_number, total_movement_pcg, vertical_acceleration, lateral_avg, state))
            data.append([vertical_displacement, int(upward_sum), int(downward_sum), state, int(total_movement_pcg), int(leftward_sum), int(rightward_sum), vertical_velocity, vertical_acceleration])

            # Update weighted masking model.
            old_weights = weights * (1 - LEARNING_RATE)
            if not is_breathing:
                weights = old_weights + preloaded_weights + ((downward_movement + upward_movement) * LEARNING_RATE)
                weights = cv2.normalize(weights, weights, 0, 1, cv2.NORM_MINMAX)
            elif state == "Compression":
                weights = weights / (1 - LEARNING_RATE)

            # Show output windows for visualization.
            if SHOW_OUTPUT:
                hsv[..., 0] = cv2.normalize(direction_in_deg, None, 0, 179, cv2.NORM_MINMAX)
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(upward_movement_weighted, None, 0, 255, cv2.NORM_MINMAX)
                flow_up_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(downward_movement_weighted, None, 0, 255, cv2.NORM_MINMAX)      
                flow_down_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                hsv[..., 2] = cv2.normalize(lateral_movement, None, 0, 255, cv2.NORM_MINMAX)      
                flow_lateral_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                weights_hsv[..., 2] = cv2.normalize(weights, None, 0, 255, cv2.NORM_MINMAX)
                weights_bgr = cv2.cvtColor(weights_hsv, cv2.COLOR_HSV2BGR)


                cv2.putText(flow_bgr, "Time: %f" % elapsed_time, (25, TEXT_START_POS_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "CCR: %fbpm" % ccr, (25, TEXT_START_POS_Y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "S: %s" % (state), (25, TEXT_START_POS_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.putText(flow_bgr, "Nc: %i" % compressions, (25, TEXT_START_POS_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * SCALE, (255,255,255), thickness=1)
                cv2.namedWindow('Solution', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (All)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (Upward)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (Downward)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Flow (Lateral)', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Weighted Mask', cv2.WINDOW_NORMAL)
                cv2.imshow('Weighted Mask', cv2.WINDOW_NORMAL)
                cv2.imshow('Solution', current_frame_bgr)
                cv2.imshow('Flow (All)', flow_bgr)
                cv2.imshow('Flow (Upward)', flow_up_bgr)
                cv2.imshow('Flow (Downward)', flow_down_bgr)
                cv2.imshow('Flow (Lateral)', flow_lateral_bgr)
                cv2.imshow('Weighted Mask', weights_bgr)

                gt_frame = imutils.rotate_bound(gt_frame, 90)
                gt_frame = cv2.resize(gt_frame, (0, 0), fx=SCALE * 0.5, fy=SCALE * 0.5) 
                cv2.namedWindow('Ground Truth', cv2.WINDOW_NORMAL)
                cv2.imshow('Ground Truth', gt_frame)
                cv2.waitKey(1)

        prev_frame_bgr = current_frame_bgr
        state = None

    VIDEO.release()
    cv2.destroyAllWindows()
    return weights, data


def main():
    gt_csv = open(GT_CSV_DIR)
    ground_truth = read_from_csv(gt_csv)

    if WEIGHTS_FILE:
        preloaded_weights = load_weights(WEIGHTS_FILE)
    else:
        preloaded_weights = None

    if RECALCULATE:
        weights, data = process_video(ground_truth, preloaded_weights)
        write_to_csv(data)
        if SAVE_WEIGHTS:
            save_weights(weights)
    else:
        try:
            existing_csv = open(CSV_DIR)
            data = read_from_csv(existing_csv)
        except FileNotFoundError:
            print("Existing solution CSV file not found for %s. Recalculating..." % DATASET) 
            weights, data = process_video(ground_truth, preloaded_weights)
            write_to_csv(data)
            if SAVE_WEIGHTS:
                save_weights(weights)

    plot_data(data, ground_truth)

if __name__ == '__main__':
    main()
