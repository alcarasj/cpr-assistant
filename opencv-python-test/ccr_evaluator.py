import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#CSV_DIR = "./csv_gt/Jerico1GT.mp4.csv"
CSV_DIR = "./csv_results/Jerico1BUV.mp4.csv"
FPS = 30

def read_from_csv(csv_file):
    """ Reads data from a CSV file. """

    print("Reading from CSV file...")
    coords = []

    reader = csv.reader(csv_file, delimiter=',')
    for index, row in enumerate(reader):
        coords.append((index + 1, row[3]))

    print("Reading from CSV file complete! %i co-ordinates read." % len(coords))
    return coords


def evaluate_ccr(data):
	""" Evaluate CCR through every datapoint, and dump the CCR for that datapoint. """

	ccr_data = np.array([])
	compressions = 0
	prev_compression_time = None
	interrupted_frames = 0

	for datapoint in data:

		elapsed_time = float(datapoint[0] / FPS)

		if datapoint[1] == "Compression":
			compressions += 1
			if prev_compression_time:
				time_diff = elapsed_time - prev_compression_time
				ccr = 60 / time_diff
				ccr_data = np.append(ccr_data, ccr)
				#print("[%i: %f] Nc: %i, CCR: %f, MEAN: %f, TIMEDIFF: %f" % (datapoint[0], elapsed_time, compressions, ccr, np.mean(ccr_data), time_diff))
			prev_compression_time = elapsed_time
		else:
			interrupted_frames += 1
	print("%s:  AVG_CCR: %f, CCF: %f, Nc: %i" % (CSV_DIR, np.mean(ccr_data), interrupted_frames / 30, compressions))

def main():
    try:
        existing_csv = open(CSV_DIR)
        data = read_from_csv(existing_csv)
        evaluate_ccr(data)
    except FileNotFoundError:
    	print("%s not found." % CSV_DIR)


if __name__ == '__main__':
    main()