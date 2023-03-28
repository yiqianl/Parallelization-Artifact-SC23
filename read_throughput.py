import csv
import sys
import os.path

# read the command line
args = sys.argv
if (len(args) <= 4):
    sys.exit('USAGE: prefix input_file out_file algo_name \n')

in_file = open(args[2], 'r')
in_lines = in_file.readlines()

prefix = args[1]
out_file = args[3]
algo_name = args[4]

runtimes = []
rows = []
one_row = ['', '']
idx = 0
counter = 0

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
            str_list = l.split(' ')
            # rows[idx].append(str_list[-1])
            # print(str_list[-1])
            # print(str_list[-1])
            one_row[0] = prefix + str_list[-1]
            # print(one_row)
        if 'Throughput' in l:
            str_list = l.split(' ')
            # print(str_list[-2])
            one_row[1] = str_list[-2]
            # print(one_row)
            csvwriter.writerow(one_row)
            counter += 1

print(counter)
