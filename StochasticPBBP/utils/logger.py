import os
import csv

class CSVLogger(object):
    def __init__(self, csv_file_name, mode='a'):
        self.csv_file_name = csv_file_name
        self.mode = mode
        pass

    def write_CSV(self, data, headers, write_header=True):
        # check all data elements are of the same length
        lengths = [len(data[key]) for key in headers]
        if len(set(lengths)) != 1:
            raise ValueError("All dictionary value lists must have the same length.")

        # make sure directory exists otherwise create
        directory = os.path.dirname(self.csv_file_name)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.csv_file_name, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row_values in zip(*(data[key] for key in headers)):
                writer.writerow(row_values)
