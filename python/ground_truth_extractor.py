import cv2
import imutils
import numpy as np
from argparse import ArgumentParser
import csv
import os
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


parser = ArgumentParser(description='Ground truth data extractor for CPR assistant test videos.')
parser.add_argument('-i', '--input', dest='input', help='Relative path to the input video file.', type=str, required=True)
parser.add_argument('-o', '--overwrite-csv', dest='RECALCULATE', help='Boolean to overwrite any existing CSV (will ignore and dump a new CSV if enabled).',  action='store_true')
parser.add_argument('-v', '--video-output', dest='video_output', help='Show video output.', action='store_true')
parser.add_argument('-d', '--debug-mode', dest='debug_mode', help='Debug mode for iterating frame-by-frame.',  action='store_true')
args = parser.parse_args()


INPUT_VIDEO = args.input
VIDEO_CAPTURE = cv2.VideoCapture("./videos/GT/%s" % INPUT_VIDEO)
FPS = int(VIDEO_CAPTURE.get(cv2.CAP_PROP_FPS))
TOTAL_FRAMES = VIDEO_CAPTURE.get(cv2.CAP_PROP_FRAME_COUNT)
RECALCULATE = args.RECALCULATE
VIDEO_OUTPUT = args.video_output
DEBUG_MODE = args.debug_mode
CSV_GT_DIR = 'csv_gt/%s.csv'

COMPRESSION_BOUNDS = (880, 900)
CALCULATE_MAXIMUMS = False or RECALCULATE 
GRAPH_AGAINST_TIME = False
print("FPS: %i" % FPS)



def show_output(frame, frame_number, y_value):
	""" Shows the video output in a window. """

	cv2.putText(frame, "Frame: %i" % frame_number, (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=3)
	cv2.putText(frame, "Y: %s" % str(y_value), (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=3)
	cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Output', 800, 800)
	cv2.imshow('Output', frame)
	if DEBUG_MODE:
		if cv2.waitKey(0) & 0xFF == ord('n'):
			pass
	else:
		cv2.waitKey(1)



def get_compressions(data):
	""" 
	Reads the raw data and gets the compressions.
	A local maximum detected within the bounds defined in COMPRESSION_BOUNDS is a compression.
	The accuracy of this method must be verified manually as there may be false positives/negatives.
	"""

	coords = data
	prev_y_value = 0
	index_of_maximum = 0
	maximum = 0
	compressions = 0
	ccr = 0
	prev_compression_time = None

	for index, value in enumerate(coords):
		current_y_value = value[1]
		if current_y_value:
			if current_y_value > prev_y_value:
				maximum = current_y_value
				index_of_maximum = index

			if current_y_value < prev_y_value and prev_y_value == maximum and COMPRESSION_BOUNDS[0] < prev_y_value and prev_y_value < COMPRESSION_BOUNDS[1]: 
				elapsed_time = float(index / FPS)
				temp = data[index_of_maximum]
				data[index_of_maximum] = [temp[0], temp[1], temp[2], "Compression"]
				compressions += 1
				if prev_compression_time:
					time_diff = elapsed_time - prev_compression_time
					ccr = 60 / time_diff
				prev_compression_time = elapsed_time
				print("[%i] Nc: %i, CCR: %i" % (index, compressions, ccr))
			prev_y_value = current_y_value
		data[index] = (value[0], value[1], value[2], None)

	return data




def write_to_csv(data):
	""" Writes the raw data (x, y) into a CSV file. """
	
	print("Writing to CSV file...")

	if os.path.exists(CSV_GT_DIR % INPUT_VIDEO):
		os.remove(CSV_GT_DIR % INPUT_VIDEO)

	try:
	    os.stat('csv_gt')
	except:
	    os.mkdir('csv_gt')
	
	csv_file = open(CSV_GT_DIR % INPUT_VIDEO, mode='w')
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	row_count = 0

	for coord in data:
		if not (coord[0] and coord[1]):
				writer.writerow(['-', '-', coord[2], 'None'])
		else:
			writer.writerow([coord[0], coord[1], coord[2], coord[3] if coord[3] else "None"])
		row_count += 1

	csv_file.close()

	print("Write to CSV file complete! %i/%i co-ordinates written." % (row_count, len(data)))



def read_from_csv(csv_file):
	""" Reads raw data (x, y) data from a CSV file. """

	print("Reading from CSV file...")
	circle_coords = []

	reader = csv.reader(csv_file, delimiter=',')
	for row in reader:
		if row[0] == '-' and row[1] == '-':
			circle_coords.append([None, None, row[2], None])
		else:
 			circle_coords.append([row[0], int(row[1]), row[2], row[3] if row[3] != 'None' else None])

	print("Reading from CSV file complete! %i co-ordinates read." % len(circle_coords))
	return circle_coords



def plot_data(data):
	""" Plots the raw data (x, y) data into a graph. """

	print("Plotting data...")
	frames = list(range(0, len(data)))
	if GRAPH_AGAINST_TIME:
		frames = [(i / FPS) for i in frames]
	y_coords = [coord[1] if not coord[3] or coord[3] == "Compression" else None for coord in data]
	compressions = [coord[1] if coord[3] == "Compression" else None for coord in data]
	breathing = [coord[1] if coord[3] == "Breathing" or coord[3] == "Compression" else None for coord in data]
	plt.plot(frames, y_coords, 'b-')
	plt.plot(frames, breathing, "b:")
	plt.plot(frames, compressions, "c*")
	plt.ylabel('Y-Coordinates of Detected Ball')
	plt.xlabel('Time (Seconds)' if GRAPH_AGAINST_TIME else 'Frame Number')
	plt.legend(['Ground Truth', 'Ground Truth (Breathing)', 'True Compression (N=%i)' % len([c for c in compressions if c])])
	plt.axis([0, frames[-1], 0, 1000])
	plt.show()



def get_raw_data():
	"""
	Reads the input video file and uses Hough circle transform to 
	detect the ball and its position.
	"""


	print("Getting raw data...")

	circle_coords = []
	complete = False

	while not complete:
		(ret, frame) = VIDEO_CAPTURE.read()
		frame_number = int(VIDEO_CAPTURE.get(cv2.CAP_PROP_POS_FRAMES))

		if ret == True:
			frame = imutils.rotate_bound(frame, 90)
			frame = cv2.medianBlur(frame, 5)

			# Hough circle transform to detect circles.
			grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			circles = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=30, minRadius=20, maxRadius=35)

			if (circles is not None) and (len(circles) == 1):
				circles = np.uint16(np.around(circles))
				ball = circles[0][0]
				coord = [ball[0], ball[1], None, None]
				# draw the outer circle
				cv2.circle(frame, (ball[0], ball[1]), ball[2], (0,255,0), 2)
				# draw the center of the circle
				cv2.circle(frame, (ball[0], ball[1]), 2, (0,0,255), 3)
			else: 
				coord = [None, None, None, None]

			circle_coords.append(coord)
			if VIDEO_OUTPUT:
				show_output(frame, frame_number, coord[1])
			prev_y_value = coord[1]
		else:
			complete = True

	VIDEO_CAPTURE.release()
	cv2.destroyAllWindows()
	print("Retrieval of raw data complete! %i co-ordinates detected." % len(circle_coords))
	return circle_coords


def main():

	if RECALCULATE:
		print("WARNING: Recalculation will overwrite the existing ground truth CSV file for %s." % INPUT_VIDEO)

	try:
		existing_csv = open(CSV_GT_DIR % INPUT_VIDEO)
		if RECALCULATE:
			raw_data = get_raw_data()
			data = get_compressions(raw_data) if CALCULATE_MAXIMUMS else raw_data
			write_to_csv(data)
		else:
			raw_data = read_from_csv(existing_csv)
			data = get_compressions(raw_data) if CALCULATE_MAXIMUMS else raw_data
			write_to_csv(data)
	except FileNotFoundError:
		print("Existing ground truth CSV file not found for %s. Recalculating..." % INPUT_VIDEO)
		raw_data = get_raw_data()
		data = get_compressions(raw_data) if CALCULATE_MAXIMUMS else raw_data
		write_to_csv(data)

	plot_data(data)
	print("Script execution complete!")


if __name__ == '__main__':
    main()
