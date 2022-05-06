#!/bin/bash

echo ""
echo "Computing results for Tables 4 & 10"
echo ""

mkdir -p results
net="7x200_best.pth" 
for dataset in mnist cifar
do
	for rep in "1" "2" "3"
	do
		method=base
		layer=base
		fn=results/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 |& tee "$fn"
		fi

		method=l_infinity
	    	for layer in "0" "1" "2" "3" "0 2" "1 2" "1 3" "1 2 3"
		do
			fn=results/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
			if test -f "$fn"; then
				echo "$fn exists; skipping."
			else
				python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
			fi
		done
	done
done

python scripts/summarize_results.py --table 4 | tee results/table4.txt
python scripts/summarize_results.py --table 10 | tee results/table10.txt



