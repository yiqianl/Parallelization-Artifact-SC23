#!bin/bash/

# convert the output to csv files
## cuda sssp
python scripts/read_throughput.py 2d throughputs/omp/sssp_2d-2e20.sym.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py co throughputs/omp/sssp_coPapersDBLP.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py rmat throughputs/omp/sssp_rmat22.sym.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py soc throughputs/omp/sssp_soc-LiveJournal1.egr_cuda.out sssp_cuda.csv sssp
python scripts/read_throughput.py ny throughputs/omp/sssp_USA-road-d.NY.egr_cuda.out sssp_cuda.csv sssp

## cuda bfs
python scripts/read_throughput.py 2d throughputs/omp/bfs_2d-2e20.sym.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py co throughputs/omp/bfs_coPapersDBLP.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py rmat throughputs/omp/bfs_rmat22.sym.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py soc throughputs/omp/bfs_soc-LiveJournal1.egr_cuda.out bfs_cuda.csv bfs
python scripts/read_throughput.py ny throughputs/omp/bfs_USA-road-d.NY.egr_cuda.out bfs_cuda.csv bfs

## cuda cc
python scripts/read_throughput.py 2d throughputs/omp/cc_2d-2e20.sym.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py co throughputs/omp/cc_coPapersDBLP.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py rmat throughputs/omp/cc_rmat22.sym.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py soc throughputs/omp/cc_soc-LiveJournal1.egr_cuda.out cc_cuda.csv cc
python scripts/read_throughput.py ny throughputs/omp/cc_USA-road-d.NY.egr_cuda.out cc_cuda.csv cc

## cuda mis
python scripts/read_throughput.py 2d throughputs/omp/mis_2d-2e20.sym.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py co throughputs/omp/mis_coPapersDBLP.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py rmat throughputs/omp/mis_rmat22.sym.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py soc throughputs/omp/mis_soc-LiveJournal1.egr_cuda.out mis_cuda.csv mis
python scripts/read_throughput.py ny throughputs/omp/mis_USA-road-d.NY.egr_cuda.out mis_cuda.csv mis

## cuda pr
python scripts/read_throughput.py 2d throughputs/omp/pr_2d-2e20.sym.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py co throughputs/omp/pr_coPapersDBLP.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py rmat throughputs/omp/pr_rmat22.sym.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py soc throughputs/omp/pr_soc-LiveJournal1.egr_cuda.out pr_cuda.csv pr
python scripts/read_throughput.py ny throughputs/omp/pr_USA-road-d.NY.egr_cuda.out pr_cuda.csv pr

## cuda tc
python scripts/read_throughput.py 2d throughputs/omp/tc_2d-2e20.sym.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py co throughputs/omp/tc_coPapersDBLP.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py rmat throughputs/omp/tc_rmat22.sym.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py soc throughputs/omp/tc_soc-LiveJournal1.egr_cuda.out tc_cuda.csv tc
python scripts/read_throughput.py ny throughputs/omp/tc_USA-road-d.NY.egr_cuda.out tc_cuda.csv t
