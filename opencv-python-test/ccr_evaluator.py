import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#CSV_DIR = "./csv_raw/Ken1FCSV.mp4FINAL.csv"
CSV_DIR = "./csv_results/Ken1BUV.mp4FINAL.csv"
FPS = 13


def read_from_csv(csv_file):
    """ Reads data from a CSV file. """

    print("Reading from CSV file...")
    coords = []

    reader = csv.reader(csv_file, delimiter=',')
    for index, row in enumerate(reader):
        coords.append((index, row[3]))

    print("Reading from CSV file complete! %i co-ordinates read." % len(coords))
    return coords


def evaluate_ccr(data):
	""" Evaluate CCR through every datapoint, and dump the CCR for that datapoint. """

	ccr_data = np.array([])
	mean_ccr = 0
	compressions = 0
	prev_compression_time = 0

	for datapoint in data:

		elapsed_time = datapoint[0] / FPS
		time_diff = elapsed_time - prev_compression_time

		if datapoint[1] == "Maximum":
			compressions += 1
			if compressions > 0:
				ccr = 60 / time_diff
				ccr_data = np.append(ccr_data, ccr)
				print("Nc: %i, CCR: %f, MEAN: %f" % (compressions, ccr, np.mean(ccr_data)))
			prev_compression_time = elapsed_time


def main():
    try:
        existing_csv = open(CSV_DIR)
        data = read_from_csv(existing_csv)
        evaluate_ccr(data)
    except FileNotFoundError:
    	print("%s not found." % CSV_DIR)


if __name__ == '__main__':
    main()