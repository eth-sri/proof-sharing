#!/bin/bash
./scripts/table1.sh
./scripts/table3.sh
./scripts/table4.sh
./scripts/table5.sh
./scripts/table6.sh
./scripts/table7.sh
./scripts/table8.sh
./scripts/table9.sh

echo ""
echo ""
echo "You are about to start the experiments for table 10, which are very expensive."
echo "To reduce the runtime, here we just us 5 samples rather than 2000, as discussed in README.md."
echo "This allows to judge whether the code works, but does not produce the results form the paper."
echo "To run the full original experiment, invoke ./script/table10.sh after removing results folder (or all fils in it stating with "linf")."
echo "Press enter to continue."
echo ""
read
./scripts/table10.sh 5
