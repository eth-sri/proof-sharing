#!/bin/bash

echo ""
echo "Computing results for Tables 5 & 11"
echo ""



mkdir -p results
dataset=mnist
net="7x200_best.pth" 
layer="1 2"
for rep in "1" "2" "3"
do
	for method in l_infinity center_and_border grid2
	do
		fn=results/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
		fi
	done
done

python scripts/summarize_results.py --table 5 | tee results/table5.txt
python scripts/summarize_results.py --table 11 | tee results/table11.txt


