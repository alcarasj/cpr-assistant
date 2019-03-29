import cv2
import numpy as np
import imutils
import os
import csv
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='CCR evaluator for ground truth and solution data.')
parser.add_argument('-d', '--dataset', dest='dataset', help='The name of the dataset e.g. Jerico1.', type=str, required=True)
args = parser.parse_args()

DATASET = args.dataset
GT_CSV_DIR = "./csv_gt/%sGT.mp4.csv" % DATASET
BUV_CSV_DIR = "./csv_results/%sBUV.mp4.csv" % DATASET
FPS = 30

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
	print("%s:  AVG_CCR: %f, CCF: %f, Nc: %i" % (DATASET, np.mean(ccr_data), interrupted_frames / 30, compressions))

def main():
    try:
        gt_csv = open(GT_CSV_DIR)
        buv_csv = open(BUV_CSV_DIR)
        gt_data = read_from_csv(gt_csv)
        buv_data = read_from_csv(buv_csv)
        evaluate_ccr(gt_data)
        evaluate_ccr(buv_data)
    except FileNotFoundError:
    	print("CSV files for %s dataset were not found." % DATASET)


if __name__ == '__main__':
    main()