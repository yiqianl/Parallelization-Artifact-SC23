#!bin/bash/

# download inputs
wget -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/coPapersDBLP.egr
wget -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/rmat22.sym.egr
wget -P inputs https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/soc-LiveJournal1.egr

# run the test cases
python run_codes.py 0 16
