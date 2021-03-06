<a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="SRI logo" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
# Shared Certificates for Neural Network Verification 

This repository contains the code for our CAV'22 paper "Shared Certificates for Neural Network Verification".
At the same time this serves as our official artifact for evaluation.

## Abstract
Existing neural network verifiers compute a proof that each input is handled correctly under a given perturbation by propagating a symbolic abstraction of reachable values at each layer. This process is repeated from scratch independently for each input (e.g., image) and perturbation (e.g., rotation), leading to an expensive overall proof effort when handling an entire dataset. In this work, we introduce a new method for reducing this verification cost without losing precision based on a key insight that abstractions obtained at intermediate layers for different inputs and perturbations can overlap or contain each other. Leveraging our insight, we introduce the general concept of shared certificates, enabling proof effort reuse across multiple inputs to reduce overall verification costs. We perform an extensive experimental evaluation to demonstrate the effectiveness of shared certificates in reducing the verification cost on a range of datasets and attack specifications on image classifiers including the popular patch and geometric perturbations.

## Organization
Here we briefly outline the structure of the repository and describe the function of individual files.
We then explain how this code can be used and how results can be replicated.
The most important files are:

```
.
├── README.md                     # this file
├── Dockerfile                    # Docker for reprudicibility/artifact
├── docker                        # scripts to run the docker
├── environment.yml               # speicfication of the conda enviornment
├── examples                      # contains data and neural networs for the examples used in evaluation
│   ├── deepg                     # code to generate geometric specifications 
│   └─── mnist_nets
│       └── templates             # template files for offline l-infinty experiments (Appendix C)
├── __main__.py                   # python interface for verification
├── check_proof_subsumption.py    # proof subsumption experiments (table 1)
├── config.py                     # default parameters for our python code
├── networks.py                   # manages neural networks
├── utils.py
├── templates.py                  # library code for proof templates (offline verification; Appendix C)
├── relaxations.py                # implementation of abstract interpretation (AI) and proof sharing
├── models.py                     # interface between neural networks and AI 
├── results                       # evaluation results are stored here
└── scripts                       # scripts to reproduce results form the paper
```

### Prerequisites
This project utilizes python (version >= 3.6), [PyTorch](http://pytorch.org) and [NumPy](http://numpy.org).
While any modern version should work, we provide exact versions for which we tested the code in `enviroment.yml`.
This file can be used to setup a [conda](https://conda.io) environment with all prerequisites:
``` bash
conda env create -f ./environment.yml
conda activate cav_proof_sharing
```
For some data generation [steps](#precomputed-results) also the [Gurobi](https://www.gurobi.com) solver is required.
While setup script installs Gurobi as required, it requires a licence. Free licences for academic use are available.

For the CAV artifact evaluation we provide a [reference setup in Docker](#Docker), that may be useful beyond the artifact evaluation.

While our code is compatible with GPU computations, here we only evaluate settings using a CPU.
Our hardware requirements are modest (any modern CPU; preferably with AVX2 and <= 4 GB of RAM), with the exception of the computation of
the optional [offline proof templates](#precomputed-results) utilized by the experiments for Appendix C. For these we recommend >= 32 GB of RAM.
This computation is not required as we provide precomputed results.

### Example Usage
We provide a command line utility for neural network verification through `__main__.py` (`python .`) for details we refer to its arguments (`python . --help`) and see the usage examples in `scripts` (see [Results](#Results)). 

### Extendability
At a high level, our code provides an implementation of common network verification algorithms as well as tools for proof sharing via templates. `__main__.py` shows multiple examples for how this library can be used for a broad range of verification tasks. The code can easily be extended to other certification problems (see behavior for different [switches](https://github.com/eth-sri/proof-sharing/blob/main/__main__.py#L92) in `__main__.py`), other neural network architectures (by adapting `network.py` and a load function as in `utils.py`) and new relaxations or proof sharing strategies by adapting (`relaxations.py` and `templates.py` respectively).

### Results
All empirical results in our paper are presented in tables. In the `scripts` folder we offer scripts to replicate all tables in the paper. (Note that, table 2 does not contain empirical results.)
Additionally we offer `all.sh`, which evaluates all tables.
Example usage:
```
./scripts/table1.sh
./scripts/tables_4_10.sh
./scripts/all.sh
```

Results are printed to screen, but also saved in `results`. For most tables the run time is between 30 min and 120 minutes. The full evaluation of Table 15 as presented in the paper takes more than 20 hours. This version evaluates 2000 samples per class. Reducing this number can greatly speed up evaluation. To this end an argument can be supplied: `./script/table15.sh <cnt>`. If no argument `<cnt>` is supplied, 2000 samples are evaluated. By default `all.sh` only runs `5` samples.

Each script states which results are in the corresponding tables are expected to be replicated exactly (by deterministic computation) and for which we expect the approximate trend to hold (due to dependence on timing, hardware, and randomness).

Note that these scripts use caching to avoid the recomputation of results. To trigger a recomputation, delete the corresponding files from `results` (or the whole folder).

#### Precomputed Results
Some of the results depends on precomputed results. In particular, for geometric specifications (tables 8, 9, 13, 14) the specifications need to be first computed. In this repository we already included these [specifications](examples/deepg/). This computation utilizes [DeepG](https://github.com/eth-sri/deepg), and we explain [here](examples/deepg/README.md) how to run it. Further, for the l-infinity experiments (Appendix C; [./scripts/table15.sh](scripts/table15.sh)) we first need to compute offline proof templates. We also provide [these](examples/mnist_nets/templates). To recompute these, run [./scripts/l_infinity_generate_5x100.sh](scripts/l_infinity_generate_5x100.sh) (runtime ~10 hours; and potentially high RAM requirements). Both of these require Gurobi (see [here](Gurobi.md)).
To benchmark these components, run the computation scripts and then (re)compute tables 8/9/13/14 and 15 respectively.

## Reproducing Results & Artifact Evaluation
In order to replicate the results from the paper simply run the scripts in the `scripts` folder.
To make setup easier, we provide a docker container with all prerequisites set up.
This is the preferred way for Artifact evaluation, but may also be useful for downstream use.

### Docker
Our docker container can build and started by running `docker build` in this folder, starting the docker container and attaching to it. The script `docker_run.sh` does this.
The minimal sequence of commands to obtain all results is:

```
sudo ./docker/docker_run.sh        # compilies and starts docker
./scripts/all.sh            # run reproduce all results tables
exit                        # stop docker & close docker
# sudo ./docker/docker_delete.sh   # optionally delete the docker container
```

As an alternative to `all.sh`, individual scripts can be run.
Note that `sudo` for the docker commands may be optional depending on you setup.


## Contributors 
- Christian Sprecher (initial author)
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc) (main contact)
- [Dimitar I. Dimitrov](https://www.sri.inf.ethz.ch/people/dimitadi)
- [Gagandeep Singh](https://ggndpsngh.github.io/)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

