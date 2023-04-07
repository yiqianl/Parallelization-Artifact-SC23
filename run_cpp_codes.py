#!/usr/bin/python3 -u

import os
import sys

cuda_path = ['codes/cuda/sssp-cuda/', 'codes/cuda/bfs-cuda/', 'codes/cuda/cc-cuda/', 'codes/cuda/mis-cuda/', 'codes/cuda/pr-cuda/', 'codes/cuda/tc-cuda/']
omp_path = ['codes/omp/sssp-omp/', 'codes/omp/bfs-omp/', 'codes/omp/cc-omp/', 'codes/omp/mis-omp/', 'codes/omp/pr-omp/', 'codes/omp/tc-omp/']
cpp_path = ['codes/cpp/sssp-cpp/', 'codes/cpp/bfs-cpp/', 'codes/cpp/cc-cpp/', 'codes/cpp/mis-cpp/', 'codes/cpp/pr-cpp/', 'codes/cpp/tc-cpp/']
source = '0'
algorithms = ['sssp', 'bfs', 'cc', 'mis', 'pr', 'tc']
inputs_folder = 'inputs/'
graph_names = ['2d-2e20.sym.egr', 'coPapersDBLP.egr', 'rmat22.sym.egr', 'soc-LiveJournal1.egr', 'USA-road-d.NY.egr']
total_num_cpp = 176
counter = 0

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) < 2):
        sys.exit('USAGE: verify thread_count(optional)\n')
    verify = args_val[1]
    thread_count = args_val[2]
    
    for graph in graph_names:
        input_path = inputs_folder + graph

        # run cpp programs
        for i in range(len(cpp_path)):
            code_path = cpp_path[i]
            out_name = algorithms[i] + '_' + graph + '_cpp.out'
            walk_code = os.walk(code_path)
            

            for root, dircs, code_files in walk_code:
                for code_file in code_files:
                    if code_file.endswith('.cpp'):
                        counter += 1
                        print("running %s, %d out of %d programs, %d out of %d inputs" % (code_file, counter, total_num_cpp, graph_names.index(graph) + 1, len(graph_names)))
                        with open(out_name, 'a') as f:
                            file_path = os.path.join(code_path, code_file)
                            sys.stdout.flush()
                            f.write('\ncompile : %s\n' % code_file)
                            sys.stdout.flush()
                            os.system('g++ %s -O3 -pthread -Iindigo_include -o minibenchmark' % file_path)
                            if 'sssp' in code_path or 'bfs' in code_path:
                                os.system('./minibenchmark %s %s %s %s >> %s' % (input_path, source, verify, thread_count, out_name))
                            elif 'pr' in code_path:
                                os.system('./minibenchmark %s %s >> %s' % (input_path, thread_count, out_name))
                            else:
                                os.system('./minibenchmark %s %s %s >> %s' % (input_path, verify, thread_count, out_name))
                            sys.stdout.flush()
                            if os.path.isfile('minibenchmark'):
                                os.system('rm minibenchmark')
                            else:
                                sys.exit('Error: compile failed')
                    else:
                        sys.exit('No C++ threads codes in the directory.')
