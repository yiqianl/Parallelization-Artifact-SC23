#!bin/bash/

# convert the output to csv files
## cuda sssp
python scripts/read_throughput.py 2d throughputs/cuda/sssp_2d-2e20.sym.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py co throughputs/cuda/sssp_coPapersDBLP.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py rmat throughputs/cuda/sssp_rmat22.sym.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py soc throughputs/cuda/sssp_soc-LiveJournal1.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py ny throughputs/cuda/sssp_USA-road-d.NY.egr_cuda.out sssp_cuda.csv sssp

## cuda bfs
python scripts/read_throughput.py 2d throughputs/cuda/bfs_2d-2e20.sym.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py co throughputs/cuda/bfs_coPapersDBLP.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py rmat throughputs/cuda/bfs_rmat22.sym.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py soc throughputs/cuda/bfs_soc-LiveJournal1.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py ny throughputs/cuda/bfs_USA-road-d.NY.egr_cuda.out bfs_cuda.csv bfs

## cuda cc
python scripts/read_throughput.py 2d throughputs/cuda/cc_2d-2e20.sym.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py co throughputs/cuda/cc_coPapersDBLP.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py rmat throughputs/cuda/cc_rmat22.sym.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py soc throughputs/cuda/cc_soc-LiveJournal1.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py ny throughputs/cuda/cc_USA-road-d.NY.egr_cuda.out cc_cuda.csv cc

## cuda mis
python scripts/read_throughput.py 2d throughputs/cuda/mis_2d-2e20.sym.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py co throughputs/cuda/mis_coPapersDBLP.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py rmat throughputs/cuda/mis_rmat22.sym.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py soc throughputs/cuda/mis_soc-LiveJournal1.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py ny throughputs/cuda/mis_USA-road-d.NY.egr_cuda.out mis_cuda.csv mis

## cuda pr
python scripts/read_throughput.py 2d throughputs/cuda/pr_2d-2e20.sym.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py co throughputs/cuda/pr_coPapersDBLP.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py rmat throughputs/cuda/pr_rmat22.sym.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py soc throughputs/cuda/pr_soc-LiveJournal1.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py ny throughputs/cuda/pr_USA-road-d.NY.egr_cuda.out pr_cuda.csv pr

## cuda tc
python scripts/read_throughput.py 2d throughputs/cuda/tc_2d-2e20.sym.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py co throughputs/cuda/tc_coPapersDBLP.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py rmat throughputs/cuda/tc_rmat22.sym.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py soc throughputs/cuda/tc_soc-LiveJournal1.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py ny throughputs/cuda/tc_USA-road-d.NY.egr_cuda.out tc_cuda.csv t
