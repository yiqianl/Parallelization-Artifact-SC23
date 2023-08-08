#!bin/bash/

# convert the output to csv files
## cpp sssp
python scripts/read_throughput.py 2d throughputs/cpp/sssp_2d-2e20.sym.egr_cpp.out data/sssp_cpp.csv sssp
python scripts/read_throughput.py co throughputs/cpp/sssp_coPapersDBLP.egr_cpp.out data/sssp_cpp.csv sssp
python scripts/read_throughput.py rmat throughputs/cpp/sssp_rmat22.sym.egr_cpp.out data/sssp_cpp.csv sssp
python scripts/read_throughput.py soc throughputs/cpp/sssp_soc-LiveJournal1.egr_cpp.out data/sssp_cpp.csv sssp
python scripts/read_throughput.py ny throughputs/cpp/sssp_USA-road-d.NY.egr_cpp.out data/sssp_cpp.csv sssp

## cpp bfs
python scripts/read_throughput.py 2d throughputs/cpp/bfs_2d-2e20.sym.egr_cpp.out data/bfs_cpp.csv bfs
python scripts/read_throughput.py co throughputs/cpp/bfs_coPapersDBLP.egr_cpp.out data/bfs_cpp.csv bfs
python scripts/read_throughput.py rmat throughputs/cpp/bfs_rmat22.sym.egr_cpp.out data/bfs_cpp.csv bfs
python scripts/read_throughput.py soc throughputs/cpp/bfs_soc-LiveJournal1.egr_cpp.out data/bfs_cpp.csv bfs
python scripts/read_throughput.py ny throughputs/cpp/bfs_USA-road-d.NY.egr_cpp.out data/bfs_cpp.csv bfs

## cpp cc
python scripts/read_throughput.py 2d throughputs/cpp/cc_2d-2e20.sym.egr_cpp.out data/cc_cpp.csv cc
python scripts/read_throughput.py co throughputs/cpp/cc_coPapersDBLP.egr_cpp.out data/cc_cpp.csv cc
python scripts/read_throughput.py rmat throughputs/cpp/cc_rmat22.sym.egr_cpp.out data/cc_cpp.csv cc
python scripts/read_throughput.py soc throughputs/cpp/cc_soc-LiveJournal1.egr_cpp.out data/cc_cpp.csv cc
python scripts/read_throughput.py ny throughputs/cpp/cc_USA-road-d.NY.egr_cpp.out data/cc_cpp.csv cc

## cpp mis
python scripts/read_throughput.py 2d throughputs/cpp/mis_2d-2e20.sym.egr_cpp.out data/mis_cpp.csv mis
python scripts/read_throughput.py co throughputs/cpp/mis_coPapersDBLP.egr_cpp.out data/mis_cpp.csv mis
python scripts/read_throughput.py rmat throughputs/cpp/mis_rmat22.sym.egr_cpp.out data/mis_cpp.csv mis
python scripts/read_throughput.py soc throughputs/cpp/mis_soc-LiveJournal1.egr_cpp.out data/mis_cpp.csv mis
python scripts/read_throughput.py ny throughputs/cpp/mis_USA-road-d.NY.egr_cpp.out data/mis_cpp.csv mis

## cpp pr
python scripts/read_throughput.py 2d throughputs/cpp/pr_2d-2e20.sym.egr_cpp.out data/pr_cpp.csv pr
python scripts/read_throughput.py co throughputs/cpp/pr_coPapersDBLP.egr_cpp.out data/pr_cpp.csv pr
python scripts/read_throughput.py rmat throughputs/cpp/pr_rmat22.sym.egr_cpp.out data/pr_cpp.csv pr
python scripts/read_throughput.py soc throughputs/cpp/pr_soc-LiveJournal1.egr_cpp.out data/pr_cpp.csv pr
python scripts/read_throughput.py ny throughputs/cpp/pr_USA-road-d.NY.egr_cpp.out data/pr_cpp.csv pr

## cpp tc
python scripts/read_throughput.py 2d throughputs/cpp/tc_2d-2e20.sym.egr_cpp.out data/tc_cpp.csv tc
python scripts/read_throughput.py co throughputs/cpp/tc_coPapersDBLP.egr_cpp.out data/tc_cpp.csv tc
python scripts/read_throughput.py rmat throughputs/cpp/tc_rmat22.sym.egr_cpp.out data/tc_cpp.csv tc
python scripts/read_throughput.py soc throughputs/cpp/tc_soc-LiveJournal1.egr_cpp.out data/tc_cpp.csv tc
python scripts/read_throughput.py ny throughputs/cpp/tc_USA-road-d.NY.egr_cpp.out data/tc_cpp.csv tc
