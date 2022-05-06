## Geometric Specifications

For some of our evaluations we use geometric specifications. 
These are specified in the subfolders of this folder. From these specifications (`*/config.txt`) actual input specifications for geometric verification can be derived. Here these are included. In order to rerun them, a patched version of [DeepG](https://github.com/eth-sri/deepg) is required.
This requires a Gurobi licence. See our recommendations [here](Gurobi.md).

To recompute the geometric specifications, run from this path:
```
./install_deepg.sh    # patch and compile deepg
./run_deepg.sh        # create speifications.
```

