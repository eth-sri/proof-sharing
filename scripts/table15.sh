#!/bin/bash

echo ""
echo "Computing results for Table 15"
echo ""

dataset=mnist
if [ -n "$1" ]; then
	nt=$1
else
	nt=2000
fi

for rep in 1 2 3
do
	for epsilon in 0.05 0.10
	do
	    for label in 0 1 2 3 4 5 6 7 8 9
	    do

		fn=results/linf${epsilon}_${dataset}${label}_5x100_base_base_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -l --netname 5x100_DiffAI.pyt --epsilon ${epsilon} --label ${label} --dataset mnist --num_tests $nt --relu_transformer box > "$fn" 2>&1
		fi

		for num_templates in 1 3 25
		do
			for layer in "2" "3" "2 3"
			do
				# Without halfspace-constraints without widening
				fn=results/linf${epsilon}_${dataset}${label}_5x100_${layer// /+}_box${num_templates}_${rep}.txt
				if test -f "$fn"; then
					echo "$fn exists; skipping."
				else
					if [[ "$layer" == "2" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_try5_s3.pkl"
					elif [[ "$layer" == "3" ]]; then
						template="5x100_DiffAI_templates_9_${label}_box_exact_try5_s3.pkl"
					elif [[ "$layer" == "2 3" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_try5_s3.pkl 5x100_DiffAI_templates_9_${label}_box_exact_try5_s3.pkl"
					fi
					python . -l --netname 5x100_DiffAI.pyt --epsilon ${epsilon} --template_layers ${layer} --label ${label} --dataset mnist --num_tests $nt --relu_transformer box --num_templates ${num_templates} --template_dir ${template} > "$fn" 2>&1
				fi

				# Without halfspace-constraints with widening
				fn=results/linf${epsilon}_${dataset}${label}_5x100_${layer// /+}_boxTE${num_templates}_${rep}.txt
				if test -f "$fn"; then
					echo "$fn exists; skipping."
				else
					if [[ "$layer" == "2" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_try5_w3.pkl"
					elif [[ "$layer" == "3" ]]; then
						template="5x100_DiffAI_templates_9_${label}_box_exact_try5_w3.pkl"
					elif [[ "$layer" == "2 3" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_try5_w3.pkl 5x100_DiffAI_templates_9_${label}_box_exact_try5_w3.pkl"
					fi
					python . -l --netname 5x100_DiffAI.pyt --epsilon ${epsilon} --template_layers ${layer}  --label ${label} --dataset mnist --num_tests $nt --relu_transformer box --num_templates ${num_templates} --template_dir ${template}  > "$fn" 2>&1
				fi

				# With halfspace-constraints without widening
				fn=results/linf${epsilon}_${dataset}${label}_5x100_${layer// /+}_star${num_templates}_${rep}.txt
				if test -f "$fn"; then
					echo "$fn exists; skipping."
				else
					if [[ "$layer" == "2" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_s3.pkl"
					elif [[ "$layer" == "3" ]]; then
						template="5x100_DiffAI_templates_9_${label}_box_exact_s3.pkl"
					elif [[ "$layer" == "2 3" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_s3.pkl 5x100_DiffAI_templates_9_${label}_box_exact_s3.pkl"
					fi
					python . -l --netname 5x100_DiffAI.pyt --epsilon ${epsilon} --template_layers ${layer}  --label ${label} --dataset mnist --num_tests $nt --relu_transformer box --num_templates ${num_templates} --template_dir ${template} > "$fn" 2>&1
				fi

				# With halfspace-constraints with widening
				fn=results/linf${epsilon}_${dataset}${label}_5x100_${layer// /+}_starTE${num_templates}_${rep}.txt
				if test -f "$fn"; then
					echo "$fn exists; skipping."
				else
					if [[ "$layer" == "2" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_w3.pkl"
					elif [[ "$layer" == "3" ]]; then
						template="5x100_DiffAI_templates_9_${label}_box_exact_w3.pkl"
					elif [[ "$layer" == "2 3" ]]; then
						template="5x100_DiffAI_templates_7_${label}_box_exact_w3.pkl 5x100_DiffAI_templates_9_${label}_box_exact_w3.pkl"
					fi
					python . -l --netname 5x100_DiffAI.pyt --epsilon ${epsilon} --template_layers ${layer}  --label ${label} --dataset mnist --num_tests $nt --relu_transformer box --num_templates ${num_templates} --template_dir ${template} > "$fn" 2>&1
				fi
			done


		done
	    done
	done
done

python scripts/summarize_results.py --table 15 | tee results/table15.txt
