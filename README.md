[![All Contributors](https://img.shields.io/badge/all_contributors-54-orange.svg?style=flat-square)](#contributors-)
<div align="center">
    <img src="docs/assets/contributors-table-small.png" alt="All Contributors Table Screenshot" width="800px" />
</div>

# Spiking Network Examples in Julia
- [Description and Motivation](#Description)
- [Example Outputs](#Example-Outputs)
- [Getting started](#getting-started)
	- [Install the Julia module](#Install-the-Julia-module)


## Description
A Network and single cell spiking neuron optimizer written in Julia.
### Motivation
[Previous work](https://github.com/russelljjarvis/BluePyOpt/blob/neuronunit_reduced_cells/examples/neuronunit/OptimizationMulitSpikingIzhikevichModel.ipynb) in data-driven optimization of spiking neurons was implemented in Python. The Python implementation of reduced model simulation sometimes called external simulation, and overall my previous implementation of reduced model optimization was slower and more complex than it needed to be, for language and tool specific reasons.

Reduced model spiking neurons models have compact equations, and they should be fast to simulate, but Python often calls external codes and programes (C,C++,NEURON,brian2,NEST,PyNN) to achieve a speedup for network simulations, however, approaches for speeding up network simulations are not necessarily efficient or convenient for running single-cell simulations, as me be required for single cell optimizations.  This strategy of calling external code causes an intolerable code complexity and intolerable run-time cost for single neuron simulations. The Python tool numba JIT partially remedies this problem, however, code from the Python optimization framework DEAP/BluePyOpt also induces an additional overhead. An almost pure Julia SNN optimization routine is a better solution to efficiently optimizing Reduced SNN models. In this package, two other packages: Evolutionary.jl, and Metaheuristics provide genetic algorithms used to optimize spiking neural networks.
 
The loss function is constructed by computing Spike Distance between all pairs of neurons
Networks are optimized using pair wise spike-distance metric on each pair of neurons
Pythons NetworkUnit package is used to perform a posthoc evaluation of the optimized network.


### Example Outputs
See the figure below where local variation and firing rates are compared against every neuron between two model networks.

#### Network optimization
For example this is a ground truth model versus an optimized model t-test of firing rates:
```
Student's t-test
	datasize: 200 	 200
	t = 11.811 	 p value = 1.82e-25
```
#### Single Cell optimization
Note for perspective 86% of spike times are matched in some of the best, model fitting competitions.
![https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/img/single_cell_spike_time_fit.png](https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/img/single_cell_spike_time_fit.png)


### Getting Started
# Install the Julia module
This is not yet an official package, so the package would need to be added in developer mode:
```
git clone https://github.com/russelljjarvis/SpikeNetOpt.jl
```

```
julia
using Pkg;
Pkg.activate(".")
or Pkg.develop()
```
### Development Plans
# DONE

- [x] Used spike distance and genetic algorithms to optimize network spike raster activity.
- [x] Used pythons NetworkUnit to validate results
- [x] NetworkUnit t-tests of results
- [x] Created single cell model fitting to Allen Brain Observatory Spike Train Data.
- [x] Implemented multi-threading
## TODO
- [ ] Optimize against FPGA Event Stream Data AEDAT4
- [ ] Implemented multi-processing
- [ ] Animation of Genetic Algorithm Convergence.
- [ ] Different Spiking Neural Network Backends (WaspNet.jl,SpikingNN.jl)
- [ ] Multiprocessing as opposed to multi-threading
- [ ] NeuroEvolution
- [ ] ADAM-Opt predictions using evolved population.
