import csv


def parseCSV(path):
    di = {}
    with open(path) as file_handle:
        file_reader = csv.reader(file_handle)
        for row in file_reader:
            features = [row[3], row[4]]
            di[row[0]] = features
    return di

f_r = parseCSV("nutrition.csv")

print(f_r)
print(len(f_r))
