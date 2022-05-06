#!/bin/bash

echo ""
echo "Computing results for Table 7"
echo ""

mkdir -p results
rep=1
method=l_infinity
net="7x200_best.pth" 
dataset=mnist
for layer in "0" "1" "2" "3"
do
	fn=results/patchestiming_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
	if test -f "$fn"; then
		echo "$fn exists; skipping."
	else
		python . -pt --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
	fi
done

python scripts/summarize_results.py --table 7 | tee results/table7.txt

