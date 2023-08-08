#!bin/bash/

# convert the output to csv files
## omp sssp
python scripts/read_throughput.py 2d throughputs/omp/sssp_2d-2e20.sym.egr_omp.out data/sssp_omp.csv sssp
python scripts/read_throughput.py co throughputs/omp/sssp_coPapersDBLP.egr_omp.out data/sssp_omp.csv sssp
python scripts/read_throughput.py rmat throughputs/omp/sssp_rmat22.sym.egr_omp.out data/sssp_omp.csv sssp
python scripts/read_throughput.py soc sssp_soc-LiveJournal1.egr_omp.out data/sssp_omp.csv sssp
python scripts/read_throughput.py ny sssp_USA-road-d.NY.egr_omp.out data/sssp_omp.csv sssp

## omp bfs
python scripts/read_throughput.py 2d bfs_2d-2e20.sym.egr_omp.out data/bfs_omp.csv bfs
python scripts/read_throughput.py co bfs_coPapersDBLP.egr_omp.out data/bfs_omp.csv bfs
python scripts/read_throughput.py rmat bfs_rmat22.sym.egr_omp.out data/bfs_omp.csv bfs
python scripts/read_throughput.py soc bfs_soc-LiveJournal1.egr_omp.out data/bfs_omp.csv bfs
python scripts/read_throughput.py ny bfs_USA-road-d.NY.egr_omp.out data/bfs_omp.csv bfs

## omp cc
python scripts/read_throughput.py 2d cc_2d-2e20.sym.egr_omp.out data/cc_omp.csv cc
python scripts/read_throughput.py co cc_coPapersDBLP.egr_omp.out data/cc_omp.csv cc
python scripts/read_throughput.py rmat cc_rmat22.sym.egr_omp.out data/cc_omp.csv cc
python scripts/read_throughput.py soc cc_soc-LiveJournal1.egr_omp.out data/cc_omp.csv cc
python scripts/read_throughput.py ny cc_USA-road-d.NY.egr_omp.out data/cc_omp.csv cc

## omp mis
python scripts/read_throughput.py 2d mis_2d-2e20.sym.egr_omp.out data/mis_omp.csv mis
python scripts/read_throughput.py co mis_coPapersDBLP.egr_omp.out data/mis_omp.csv mis
python scripts/read_throughput.py rmat mis_rmat22.sym.egr_omp.out data/mis_omp.csv mis
python scripts/read_throughput.py soc mis_soc-LiveJournal1.egr_omp.out data/mis_omp.csv mis
python scripts/read_throughput.py ny mis_USA-road-d.NY.egr_omp.out data/mis_omp.csv mis

## omp pr
python scripts/read_throughput.py 2d pr_2d-2e20.sym.egr_omp.out data/pr_omp.csv pr
python scripts/read_throughput.py co pr_coPapersDBLP.egr_omp.out data/pr_omp.csv pr
python scripts/read_throughput.py rmat pr_rmat22.sym.egr_omp.out data/pr_omp.csv pr
python scripts/read_throughput.py soc pr_soc-LiveJournal1.egr_omp.out data/pr_omp.csv pr
python scripts/read_throughput.py ny pr_USA-road-d.NY.egr_omp.out data/pr_omp.csv pr

## omp tc
python scripts/read_throughput.py 2d tc_2d-2e20.sym.egr_omp.out data/tc_omp.csv tc
python scripts/read_throughput.py co tc_coPapersDBLP.egr_omp.out data/tc_omp.csv tc
python scripts/read_throughput.py rmat tc_rmat22.sym.egr_omp.out data/tc_omp.csv tc
python scripts/read_throughput.py soc tc_soc-LiveJournal1.egr_omp.out data/tc_omp.csv tc
python scripts/read_throughput.py ny tc_USA-road-d.NY.egr_omp.out data/tc_omp.csv tc
