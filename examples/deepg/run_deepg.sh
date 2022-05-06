#!/bin/bash

for folder in mnist_1_brightness_01_001_proof_transfer10 mnist_1_brightness_01_001_proof_transfer4 mnist_1_brightness_01_001_proof_transfer6 mnist_1_brightness_01_001_proof_transfer8 mnist_1_rotation_40_proof_transfer
do
	echo "Creating deepg specifications for "$folder
	echo "  deleting old specifications"
	find $folder ! -name 'config.txt' -type f -exec rm -f {} +

	cd deepg/code
	LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ./build/deepg_constraints ../../$folder
	cd ../..

done
