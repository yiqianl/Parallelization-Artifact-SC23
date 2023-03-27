import csv
import sys
import os.path


# read the command line
args = sys.argv
if (len(args) <= 3):
    sys.exit('USAGE: graph_name input_file out_file algo_name\n')

rows = []
graph_name = args[1]
out_file = args[3]
algo_name = args[4]

with open(args[2]) as file_obj:

    # Skips the heading
    # Using next() method
    heading = next(file_obj)

    # Create reader object by passing the file
    # object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        row[0] = graph_name + row[0]
        # print(row)
        rows.append(row)
    # print(rows)





runtimes = []
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


    for row in rows:
        csvwriter.writerow(row)
        # if 'compile' in l:
        #     str_list = l.split(' ')
        #     # rows[idx].append(str_list[-1])
        #     # print(str_list[-1])
        #     # print(str_list[-1])
        #     one_row[0] = str_list[-1]
        #     # print(one_row)
        # if 'Throughput' in l:
        #     str_list = l.split(' ')
        #     # print(str_list[-2])
        #     one_row[1] = str_list[-2]
        #     # print(one_row)

        #     counter += 1

# print(counter)
