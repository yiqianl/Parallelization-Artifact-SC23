#!bin/bash/

# convert the output to csv files
## cpp sssp
python scripts/read_throughput.py soc throughputs/cpp/sssp_soc-LiveJournal1.egr_cpp.out data/sssp_cpp.csv sssp

## cpp bfs
python scripts/read_throughput.py soc throughputs/cpp/bfs_soc-LiveJournal1.egr_cpp.out data/bfs_cpp.csv bfs

## cpp cc
python scripts/read_throughput.py soc throughputs/cpp/cc_soc-LiveJournal1.egr_cpp.out data/cc_cpp.csv cc

## cpp mis
python scripts/read_throughput.py soc throughputs/cpp/mis_soc-LiveJournal1.egr_cpp.out data/mis_cpp.csv mis

## cpp pr
python scripts/read_throughput.py soc throughputs/cpp/pr_soc-LiveJournal1.egr_cpp.out data/pr_cpp.csv pr

## cpp tc
python scripts/read_throughput.py soc throughputs/cpp/tc_soc-LiveJournal1.egr_cpp.out data/tc_cpp.csv tc
