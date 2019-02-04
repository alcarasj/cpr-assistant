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
parser.add_argument('-o', '--overwrite-csv', dest='overwrite_csv', help='Boolean to overwrite any existing CSV (will ignore and dump a new CSV).', type=bool, required=False, default=False)
parser.add_argument('-v', '--video-output', dest='video_output', help='Show video output.', type=bool, required=False, default=False)
args = parser.parse_args()


INPUT_VIDEO = args.input
VIDEO_CAPTURE = cv2.VideoCapture(INPUT_VIDEO)
TOTAL_FRAMES = VIDEO_CAPTURE.get(cv2.CAP_PROP_FRAME_COUNT)
OVERWRITE_CSV = args.overwrite_csv
START_FRAME = args.start_frame
VIDEO_OUTPUT = args.video_output
CSV_RAW_DIR = 'csv_raw/%s.csv'

def show_output(frame):
	cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Output', 800, 800)
	cv2.imshow('Output', frame)
	cv2.waitKey(1) 


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
			writer.writerow(['-', '-', coord[2]])
		else:
			writer.writerow([coord[0], coord[1], coord[2]])
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
			circle_coords.append((None, None))
		else:
 			circle_coords.append((row[0], row[1]))

	print("Reading from CSV file complete! %i co-ordinates read." % len(circle_coords))
	return circle_coords



def plot_data(data):
	""" Plots the raw data (x, y) data into a graph. """

	print("Plotting data...")
	frames = list(range(1, len(data) + 1))
	y_coords = list(map(lambda coord: coord[1], data))
	plt.plot(frames, y_coords)
	plt.ylabel('Y-Coordinates')
	plt.xlabel('Frame')
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
				# Output windows. 
				# cv2.putText(grayscale, str(frame_number), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
				# cv2.putText(grayscale, str(avg_value), (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=5)
				# draw the outer circle
				cv2.circle(frame, (ball[0], ball[1]), ball[2], (0,255,0), 2)
				# draw the center of the circle
				cv2.circle(frame, (ball[0], ball[1]), 2, (0,0,255), 3)
			else: 
				coord = (None, None, avg_value)

			circle_coords.append(coord)
			if VIDEO_OUTPUT:
				show_output(frame)
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
			write_to_csv(raw_data)
		else:
			raw_data = read_from_csv(existing_csv)
	except FileNotFoundError:
		raw_data = get_raw_data()
		write_to_csv(raw_data)

	plot_data(raw_data)
	print("Script execution complete!")


if __name__ == '__main__':
    main()
