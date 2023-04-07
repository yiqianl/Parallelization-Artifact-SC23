# convert the output to csv files

## cpp sssp
python read_throughput.py 2d sssp_2d-2e20.sym.egr_cpp.out sssp_cpp.csv sssp
python read_throughput.py co sssp_coPapersDBLP.egr_cpp.out sssp_cpp.csv sssp
python read_throughput.py rmat sssp_rmat22.sym.egr_cpp.out sssp_cpp.csv sssp
python read_throughput.py soc sssp_soc-LiveJournal1.egr_cpp.out sssp_cpp.csv sssp
python read_throughput.py ny sssp_USA-road-d.NY.egr_cpp.out sssp_cpp.csv sssp

## cpp bfs
python read_throughput.py 2d bfs_2d-2e20.sym.egr_cpp.out bfs_cpp.csv bfs
python read_throughput.py co bfs_coPapersDBLP.egr_cpp.out bfs_cpp.csv bfs
python read_throughput.py rmat bfs_rmat22.sym.egr_cpp.out bfs_cpp.csv bfs
python read_throughput.py soc bfs_soc-LiveJournal1.egr_cpp.out bfs_cpp.csv bfs
python read_throughput.py ny bfs_USA-road-d.NY.egr_cpp.out bfs_cpp.csv bfs

## cpp cc
python read_throughput.py 2d cc_2d-2e20.sym.egr_cpp.out cc_cpp.csv cc
python read_throughput.py co cc_coPapersDBLP.egr_cpp.out cc_cpp.csv cc
python read_throughput.py rmat cc_rmat22.sym.egr_cpp.out cc_cpp.csv cc
python read_throughput.py soc cc_soc-LiveJournal1.egr_cpp.out cc_cpp.csv cc
python read_throughput.py ny cc_USA-road-d.NY.egr_cpp.out cc_cpp.csv cc

## cpp mis
python read_throughput.py 2d mis_2d-2e20.sym.egr_cpp.out mis_cpp.csv mis
python read_throughput.py co mis_coPapersDBLP.egr_cpp.out mis_cpp.csv mis
python read_throughput.py rmat mis_rmat22.sym.egr_cpp.out mis_cpp.csv mis
python read_throughput.py soc mis_soc-LiveJournal1.egr_cpp.out mis_cpp.csv mis
python read_throughput.py ny mis_USA-road-d.NY.egr_cpp.out mis_cpp.csv mis

## cpp pr
python read_throughput.py 2d pr_2d-2e20.sym.egr_cpp.out pr_cpp.csv pr
python read_throughput.py co pr_coPapersDBLP.egr_cpp.out pr_cpp.csv pr
python read_throughput.py rmat pr_rmat22.sym.egr_cpp.out pr_cpp.csv pr
python read_throughput.py soc pr_soc-LiveJournal1.egr_cpp.out pr_cpp.csv pr
python read_throughput.py ny pr_USA-road-d.NY.egr_cpp.out pr_cpp.csv pr

## cpp tc
python read_throughput.py 2d tc_2d-2e20.sym.egr_cpp.out tc_cpp.csv tc
python read_throughput.py co tc_coPapersDBLP.egr_cpp.out tc_cpp.csv tc
python read_throughput.py rmat tc_rmat22.sym.egr_cpp.out tc_cpp.csv tc
python read_throughput.py soc tc_soc-LiveJournal1.egr_cpp.out tc_cpp.csv tc
python read_throughput.py ny tc_USA-road-d.NY.egr_cpp.out tc_cpp.csv tc