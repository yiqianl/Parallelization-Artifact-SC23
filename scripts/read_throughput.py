import csv
import sys
import os.path

# read the command line
args = sys.argv
if (len(args) <= 4):
    sys.exit('USAGE: prefix input_file out_file algo_name \n')

input_file_path = args[2]
in_file = open(input_file_path, 'r')
in_lines = in_file.readlines()

prefix = args[1]
out_file = args[3]
algo_name = args[4]

runtimes = []
rows = []
one_row = ['', '']
idx1 = 0
idx2 = 0

header = True
if os.path.isfile(out_file):
    file = open(out_file, 'a')
    header = False
else:
    file = open(out_file, 'w')

with file as csvfile:
    csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    if header:
        csvwriter.writerow(['filename', algo_name])


    for l in in_lines:
        if 'compile' in l:
            # str_list = l.split(' ')
            # one_row[0] = (prefix + str_list[-1]).replace('\n', '')
            # rows.append(one_row)
            if idx1 == len(rows):
                rows.append(['', ''])
            rows[idx1][0] = (prefix + l.split(' ')[-1]).replace('\n', '')
            idx1 += 1
        if 'Throughput' in l:
            # str_list = l.split(' ')
            # rows[idx][1] = (str_list[-2])
            # csvwriter.writerow(rows[idx])
            # idx += 1
            if idx2 == len(rows):
                rows.append(['', ''])
            rows[idx2][1] = (l.split(' ')[-2])
            idx2 += 1

    for row in rows:
        csvwriter.writerow(row)

# print(idx)
