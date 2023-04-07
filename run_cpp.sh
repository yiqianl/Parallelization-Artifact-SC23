#!bin/bash/

# download inputs
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/2d-2e20.sym.egr
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/coPapersDBLP.egr
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/rmat22.sym.egr
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/soc-LiveJournal1.egr
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/USA-road-d.NY.egr

# run the test cases
python run_cpp_codes.py 0 16
