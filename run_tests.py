#!/usr/bin/python3 -u

import os
import sys

input_path = 'input/power_law_36n_140e.egr'
cuda_path = ['cuda/sssp/', 'cuda/bfs/', 'cuda/cc/', 'cuda/mis/', 'cuda/pr', 'cuda/tc']
omp_path = ['omp/sssp/', 'omp/bfs/', 'omp/cc/', 'omp/mis/', 'omp/pr', 'omp/tc']
cpp_path = ['cpp/sssp/', 'cpp/bfs/', 'cpp/cc/', 'cpp/mis/', 'cpp/pr', 'cpp/tc']

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) < 3):
        sys.exit('USAGE: verify thread_count(optional)\n')
    verify = args_val[1]
    thread_count = args_val[2]

    # run cuda programs
    for code_path in cuda_path:
        walk_code = os.walk(code_path)
        for root, dircs, code_files in walk_code:
            for code_file in code_files:
                if code_file.endswith('.cu'):
                    file_path = os.path.join(code_path, code_file)
                    sys.stdout.flush()
                    print('\ncompile : %s\n' % code_file)
                    sys.stdout.flush()
                    os.system('nvcc %s -O3 -arch=sm_70 -Iindigo_include -o minibenchmark' % file_path)
                    os.system('./minibenchmark %s %s' % (input_path, verify))
                    sys.stdout.flush()
                    if os.path.isfile('microbenchmark'):
                        os.system('rm microbenchmark')
                    else:
                        sys.exit('Error: compile failed')
                else:
                    sys.exit('No cuda codes in the directory.')

    # run cpp programs
    for code_path in cpp_path:
        walk_code = os.walk(code_path)
        for root, dircs, code_files in walk_code:
            for code_file in code_files:
                if code_file.endswith('.cpp'):
                    file_path = os.path.join(code_path, code_file)
                    sys.stdout.flush()
                    print('\ncompile : %s\n' % code_file)
                    sys.stdout.flush()
                    os.system('g++ %s -O3 -pthread -Iindigo_include -o minibenchmark' % file_path)
                    os.system('./minibenchmark %s %s %s' % (input_path, verify, thread_count))
                    sys.stdout.flush()
                    if os.path.isfile('microbenchmark'):
                        os.system('rm microbenchmark')
                    else:
                        sys.exit('Error: compile failed')
                else:
                    sys.exit('No C++ threads codes in the directory.')

    for code_path in omp_path:
        walk_code = os.walk(code_path)
        for root, dircs, code_files in walk_code:
            for code_file in code_files:
                if code_file.endswith('.cpp'):
                    file_path = os.path.join(code_path, code_file)
                    sys.stdout.flush()
                    print('\ncompile : %s\n' % code_file)
                    sys.stdout.flush()
                    os.system('g++ %s -O3 -fopenmp -Iindigo_include -o minibenchmark' % file_path)
                    os.system('./minibenchmark %s %s' % (input_path, verify))
                    sys.stdout.flush()
                    if os.path.isfile('microbenchmark'):
                        os.system('rm microbenchmark')
                    else:
                        sys.exit('Error: compile failed')
                else:
                    sys.exit('No omp codes in the directory.')
