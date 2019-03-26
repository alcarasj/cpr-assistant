import cv2
import imutils
import numpy as np
from argparse import ArgumentParser
import csv
import os
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


parser = ArgumentParser(description='Raw data retriever for CPR assistant test videos.')
parser.add_argument('-i', '--input', dest='input', help='Relative path to the input video file.', type=str, required=True)
parser.add_argument('-s', '--start-frame', dest='start_frame', help='The frame number of the start of compressions.', type=int, required=True)
parser.add_argument('-o', '--overwrite-csv', dest='overwrite_csv', help='Boolean to overwrite any existing CSV (will ignore and dump a new CSV if enabled).', type=bool, required=False, default=False)
parser.add_argument('-v', '--video-output', dest='video_output', help='Show video output.', type=bool, required=False, default=False)
parser.add_argument('-d', '--debug-mode', dest='debug_mode', help='Debug mode for iterating frame-by-frame.', type=bool, required=False, default=False)
args = parser.parse_args()


INPUT_VIDEO = args.input
VIDEO_CAPTURE = cv2.VideoCapture(INPUT_VIDEO)
TOTAL_FRAMES = VIDEO_CAPTURE.get(cv2.CAP_PROP_FRAME_COUNT)
OVERWRITE_CSV = args.overwrite_csv
START_FRAME = args.start_frame
VIDEO_OUTPUT = args.video_output
DEBUG_MODE = args.debug_mode
CSV_RAW_DIR = 'csv_raw/%s.csv'



def show_output(frame, frame_number, y_value, avg_value):
	""" Shows the video output in a window. """

	cv2.putText(frame, "Frame: %i" % frame_number, (150,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
	cv2.putText(frame, "Value: %i" % avg_value, (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
	cv2.putText(frame, "Y: %s" % str(y_value), (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
	cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Output', 800, 800)
	cv2.imshow('Output', frame)
	if DEBUG_MODE:
		if cv2.waitKey(0) & 0xFF == ord('n'):
			pass
	else:
		cv2.waitKey(1)



def get_minima_maxima(data):
	""" 
	Reads the raw data and gets the local minimums and maximums.
	The accuracy of this method must be verified manually.
	"""

	coords = data
	prev_y_value = 0
	last_min_index = 0
	last_max_index = 0
	prev_index = 0
	minimum = 0
	maximum = 0

	for index, value in enumerate(coords):
		current_y_value = value[1]
		if index > START_FRAME - 1 and current_y_value:
			if current_y_value < prev_y_value:
				minimum = current_y_value
				last_min_index = index
			elif current_y_value > prev_y_value:
				maximum = current_y_value
				last_max_index = index

			if current_y_value > prev_y_value and prev_y_value == minimum:
				prev_value = data[last_min_index]
				data[last_min_index] = (prev_value[0], prev_value[1], prev_value[2], "Minimum")
				print("Ymin = %i" % current_y_value)
			elif current_y_value < prev_y_value and prev_y_value == maximum:
				prev_value = data[last_max_index]
				data[last_max_index] = (prev_value[0], prev_value[1], prev_value[2], "Maximum")
				print("Ymax = %i" % current_y_value)
			print("[%i] %i" % (index, current_y_value))
			prev_y_value = current_y_value
			prev_index = index
		data[index] = (value[0], value[1], value[2], None)

	return data




def write_to_csv(data):
	""" Writes the raw data (x, y) data into a CSV file. """
	
	print("Writing to CSV file...")

	if os.path.exists(CSV_RAW_DIR % INPUT_VIDEO):
		os.remove(CSV_RAW_DIR % INPUT_VIDEO)

	try:
	    os.stat('csv_raw')
	except:
	    os.mkdir('csv_raw')
	
	csv_file = open(CSV_RAW_DIR % INPUT_VIDEO, mode='w')
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
			circle_coords.append((None, None, row[2], None))
		else:
 			circle_coords.append((row[0], int(row[1]), row[2], row[3] if row[3] != 'None' else None))

	print("Reading from CSV file complete! %i co-ordinates read." % len(circle_coords))
	return circle_coords



def plot_data(data):
	""" Plots the raw data (x, y) data into a graph. """

	print("Plotting data...")
	frames = list(range(1, len(data) + 1))
	y_coords = [coord[1] for coord in data]
	max_min = [coord[1] if coord[3] == "Maximum" else None for coord in data]
	plt.plot(frames, y_coords)
	plt.plot(frames, max_min, "x")
	plt.ylabel('Y-Coordinates of Hough Circle Transform')
	plt.xlabel('Frame')
	plt.legend(['Ground Truth', 'Compression'])
	plt.axis([0, len(data), 0, 2000])
	plt.show()



def get_raw_data():
	"""
	Reads the input video file and uses Hough circle transform to 
	retrieve raw in the form of (x, y) co-ordinates.
	"""


	print("Getting raw data...")

	circle_coords = []
	complete = False

	while not complete:
		(ret, frame) = VIDEO_CAPTURE.read()
		frame_number = int(VIDEO_CAPTURE.get(cv2.CAP_PROP_POS_FRAMES))

		in_valid_frame = (frame_number <= TOTAL_FRAMES) and (frame is not None)

		if in_valid_frame:
			frame = imutils.rotate_bound(frame, 270)
			frame = cv2.medianBlur(frame, 5)

			# HSV for brightness value to determine start of compressions.
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

			# Mean value of frame.
			avg_value = int(cv2.mean(hsv)[0])

			# Print values per frame.
			# print("[%i] - %i" % (frame_number, avg_value))

			# Hough circle transform to detect circles.
			grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			circles = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=30, minRadius=30, maxRadius=45)

			if (circles is not None) and (len(circles) == 1):
				circles = np.uint16(np.around(circles))
				ball = circles[0][0]
				coord = (ball[0], ball[1], avg_value)
				# draw the outer circle
				cv2.circle(frame, (ball[0], ball[1]), ball[2], (0,255,0), 2)
				# draw the center of the circle
				cv2.circle(frame, (ball[0], ball[1]), 2, (0,0,255), 3)
			else: 
				coord = (None, None, avg_value, None)

			circle_coords.append(coord)
			if VIDEO_OUTPUT:
				show_output(frame, frame_number, coord[1], avg_value)
			prev_y_value = coord[1]
		else:
			complete = True

	VIDEO_CAPTURE.release()
	cv2.destroyAllWindows()
	print("Retrieval of raw data complete! %i co-ordinates detected." % len(circle_coords))
	return circle_coords



def main():

	if OVERWRITE_CSV:
		print("WARNING: CSV OVERWRITE MODE.")

	try:
		existing_csv = open(CSV_RAW_DIR % INPUT_VIDEO)
		if OVERWRITE_CSV:
			raw_data = get_raw_data()
			data = get_minima_maxima(raw_data)
			write_to_csv(data)
		else:
			raw_data = read_from_csv(existing_csv)
			data = get_minima_maxima(raw_data)
			#write_to_csv(data)
	except FileNotFoundError:
		raw_data = get_raw_data()
		data = get_minima_maxima(raw_data)
		write_to_csv(data)

	plot_data(data)
	print("Script execution complete!")


if __name__ == '__main__':
    main()
