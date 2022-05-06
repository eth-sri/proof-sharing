## Gurobi Solver
For two (optional) evaluation steps we require the [Gurobi](https://www.gurobi.com) solver with a valid licence.
Both our docker and conda setup install the correct version.
In order to obtain an academic licence please refer to their website.

### Docker
If Gurobi is used through a Docker container (or VM) a special licence is required.
Such a "Web License Service for Container Environments" can be obtained [here](
https://www.gurobi.com/academia/academic-program-and-licenses).
If the docker container is running with name `proof` and the licence file `gruobi.lic` is located on the host computer it can be correctly installed via `docker cp gurobi.lic proof:/root/gurobi.lic`.
