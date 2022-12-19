#!/bin/bash

echo ""
echo "Computing results for Tables 9"
echo ""


net="7x200_best.pth" 
dataset="mnist"

for rep in "1" "2" "3"
do
	layer="base"
	method="base"
	fn=results/geometric40_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
	if test -f "$fn"; then
		echo "$fn exists; skipping."
	else
		python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_rotation_40_proof_transfer |& tee "$fn"
	fi

        for method in l_infinity rotation2_40 rotation3_40
	do
		layer="1 2"
		fn=results/geometric40_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else

            		python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_rotation_40_proof_transfer --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
		fi
	done
done

python scripts/summarize_results.py --table 9 | tee results/table9.txt
