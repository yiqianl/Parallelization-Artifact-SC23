#!bin/bash/

# download inputs
wget -N -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/soc-LiveJournal1.egr

# run the test cases
python scripts/run_one_cpp_codes.py 0 16
