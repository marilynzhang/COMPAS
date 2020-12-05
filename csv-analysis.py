import csv

def read_csv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]
    return data_read

# data[i] is info for the ith defendant; data[0] is the column headers
all_data = read_csv("compase-scores-two-years.csv")
