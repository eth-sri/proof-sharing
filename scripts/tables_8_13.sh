#!/bin/bash

echo ""
echo "Computing results for Tables 8 & 13"
echo ""


net="7x200_best.pth" 
dataset="mnist"

for rep in "1" "2" "3"
do
	for num_splits in 4 6 8 10
	do
		layer="base"
		method="base"
		fn=results/geometrics${num_splits}_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_brightness_01_001_proof_transfer${num_splits} |& tee "$fn"
		fi


		layer="1 2"
		method=l_infinity
		fn=results/geometrics${num_splits}_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_brightness_01_001_proof_transfer${num_splits} --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
		fi
	done
done

python scripts/summarize_results.py --table 8 | tee results/table8.txt
python scripts/summarize_results.py --table 13 | tee results/table13.txt
