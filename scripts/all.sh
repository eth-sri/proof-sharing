#!/bin/bash
./scripts/table1.sh
./scripts/table3.sh
./scripts/tables_4_10.sh
./scripts/tables_5_11.sh
./scripts/tables_6_12.sh
./scripts/table7.sh
./scripts/tables_8_13.sh
./scripts/tables_9_14.sh

echo ""
echo ""
echo "You are about to start the experiments for 15, which are very expensive."
echo "To reduce the runtime, here we just us 5 samples rather than 2000, as discussed in README.md."
echo "This allows to judge whether the code works, but does not produce the results form the paper."
echo "To run the full original experiment, invoke ./script/table15.sh after removing results folder (or all fils in it stating with "linf")."
echo "Press enter to continue."
echo ""
read
./scripts/table15.sh 5
