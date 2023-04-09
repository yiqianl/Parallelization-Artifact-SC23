#!/usr/bin/python3 -u

import os
import sys

cpp_path = ['codes/cpp/sssp-cpp/', 'codes/cpp/bfs-cpp/', 'codes/cpp/cc-cpp/', 'codes/cpp/mis-cpp/', 'codes/cpp/pr-cpp/', 'codes/cpp/tc-cpp/']
source = '0'
algorithms = ['sssp', 'bfs', 'cc', 'mis', 'pr', 'tc']
inputs_folder = 'inputs/'
out_dir = 'throughputs/'
cpp_out = 'throughputs/cpp/'
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

    # create output directory
    if os.path.exists(out_dir):
        os.system('rm -rf %s' % (out_dir))
    os.mkdir(out_dir)

    for graph in graph_names:
        input_path = inputs_folder + graph

        # run cpp programs
        for i in range(len(cpp_path)):
            code_path = cpp_path[i] 
            walk_code = os.walk(code_path)

            # create cpp out directory
            if os.path.exists(cpp_out):
                os.rmdir(cpp_out)
            os.mkdir(cpp_out)
            out_name = cpp_out + algorithms[i] + '_' + graph + '_cpp.out'
            
            if os.path.isfile(out_name):
                os.remove(out_name)

            for root, dircs, code_files in walk_code:
                for code_file in code_files:
                    if code_file.endswith('.cpp'):
                        counter += 1
                        sys.stdout.flush()
                        print("running %s\n%d out of %d programs, %d out of %d inputs" % (code_file, counter, total_num_cpp, graph_names.index(graph) + 1, len(graph_names)))
                        sys.stdout.flush()
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
