#!/bin/bash

echo ""
echo "Computing results for Table 1"
echo ""

mkdir -p results
echo "" | tee results/table1.txt
echo "Table 1" | tee -a results/table1.txt
echo "Expect Result: This table contains deterministically reproducible results." | tee -a results/table1.txt
echo "               The result is expected to be the same as in the paper." | tee -a results/table1.txt
echo "" | tee -a results/table1.txt

mkdir -p results
python  check_proof_subsumption.py --model 5x100_DiffAI.pyt  --eps 0.1 | tee -a results/table1.txt
echo "" | tee -a results/table1.txt
python  check_proof_subsumption.py --model 5x100_DiffAI.pyt  --eps 0.2 | tee -a results/table1.txt
