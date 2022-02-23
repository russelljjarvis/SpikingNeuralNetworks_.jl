<h1 align="center">
  Spiking Network Examples in Julia
</h1>

<p align="center">
  <img alt="GitHub" src="https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square">
  <img alt="GitHub" src="https://img.shields.io/github/issues/russelljjarvis/SpikeNetOpt.jl.svg">
  <img alt="GitHub" src="https://img.shields.io/github/issues-closed/russelljjarvis/SpikeNetOpt.jl.svg">
  <img alt="GitHub" src="https://img.shields.io/github/commit-activity/m/russelljjarvis/SpikeNetOpt.jl.svg">
  <img alt="GitHub" src="https://github.com/russelljjarvis/SpikeNetOpt.jl/workflows/CI/badge.svg">

</p>

<!--  <img alt="GitHub" src="https://github.com/russelljjarvis/SpikeNetOpt.jl/workflows/CI/badge.svg"> -->

<p align="center">
  <a href="#Description">Description</a> •
  <a href="#Example-Outputs">Outputs</a> •
  <a href="#Motivation">Motivation</a> •
  <a href="#Install-the-Julia-module">Install</a> •
  <a href="#Development-Plans">Plans</a> •
  <a href="#Current-Design-Flow-Chart">Flow Chart</a> •
  <a href="#Why-Not-Optimize-Small-SNNs-With-Bigger-SNNs">Why Not Optimize Small SNNs With Bigger SNNs</a>
</p>

<p align="center">
Julia has enough tools to support fitting spiking neural network models to data. Python speed necessitates external simulators to do network simulation. As much as possible it would be nice to do fast, efficient data fitting of spike trains to network models in one language, lets try to do that here.
</p>

### Due to Dependency Issues, 
this project is split into two repositories: [GreenerML](https://github.com/russelljjarvis/GreenerML.jl) and this one.
GreenerML uses SpikingNN.jl as the Network simulator backend, and Evolutionary.jl to optimize. This one uses SpikingNeuralNetworks.jl as the backend, and Metahieristics.jl to optimize.


### Getting Started

<details>
  <summary>Install the Julia module</summary>
  

This is not yet an official package, so the package would need to be added in developer mode. The short way to do this is as follows:
```
import Pkg
Pkg.add(url="https://github.com/russelljjarvis/SpikeNetOpt.jl.git")
```
or 
```
] add https://github.com/russelljjarvis/SpikeNetOpt.jl.git
```
The long way invovles:
```
git clone https://github.com/russelljjarvis/SpikeNetOpt.jl
```

```
cd SpikeNetOpt.jl
julia
]
(@v1.5) pkg> develop .
```
Or
```
Pkg.develop(PackageSpec(path=pwd()))

```
</details>

<details>
<summary>Entry Points</summary>



Optimize a spiking neural network by exploring effect of parameter that controls connectome graph structure:
```
julia
include("examples/run_net_opt.jl")

```
```
cd examples
julia run_net_opt.jl
```

Single cell data fitting against spike times:
```
cd test
julia single_cell_opt_adexp.jl
julia single_cell_opt_izhi.jl
```


</details>

### Motivation
<details>
  <summary>Detailed Motivation and Previous work</summary>

(https://github.com/russelljjarvis/BluePyOpt/blob/neuronunit_reduced_cells/examples/neuronunit/OptimizationMulitSpikingIzhikevichModel.ipynb) in data-driven optimization of spiking neurons was implemented in Python. The Python implementation of reduced model simulation sometimes called external simulation, and overall my previous implementation of reduced model optimization was slower and more complex than it needed to be, for language and tool specific reasons.

Reduced model spiking neurons models have compact equations, and they should be fast to simulate, but Python often calls external codes and programes (C,C++,NEURON,brian2,NEST,PyNN) to achieve a speedup for network simulations, however, approaches for speeding up network simulations are not necessarily efficient or convenient for running single-cell simulations, as me be required for single cell optimizations.  This strategy of calling external code causes an intolerable code complexity and intolerable run-time cost for single neuron simulations. The Python tool numba JIT partially remedies this problem, however, code from the Python optimization framework DEAP/BluePyOpt also induces an additional overhead. An almost pure Julia SNN optimization routine is a better solution to efficiently optimizing Reduced SNN models. In this package, two other packages: Evolutionary.jl, and Metaheuristics provide genetic algorithms used to optimize spiking neural networks.

	
</details>

#### Other Info
[A Google Doc presentation](https://docs.google.com/presentation/d/1bWA5LhgAD8D4MGPQxf5P6jtb0spVEGeJKyXCHnh-aq0/edit?usp=sharing) that sets up the motivation for the project.
[Part of BrainHack](https://brainhack.org/global2021/project/project_98/)

#### Optimization Outputs
The loss function is constructed by computing Spike Distance between all pairs of neurons
Networks are optimized using pair wise spike-distance metric on each pair of neurons
Pythons NetworkUnit package is used to perform a posthoc evaluation of the optimized network.
<details>
  <summary>Example Outputs</summary>


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
Output from a single cell optimization:

<p align="center">
	<img src="https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/img/single_cell_spike_time_fit.png" width="250" height="200">
</p>

Output from a Network Spike Time optimization (note that Unicode backend is the plotting method, and neuron synapses fire propabilistically):

<p align="center">
	<img src="https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/img/net_compare_unicode.png" width="250" height="200">
</p>
</details>

#### Current Design Vs Intended Design Flow Chart:


<p align="center">
	<img src="https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/doc/Flowchart%20(2).jpg" width="250" height="200">
</p>
<h4> Intended Future Design: </h4>
<p align="center">
	<img src="https://github.com/russelljjarvis/SpikeNetOpt.jl/blob/main/img/second_flow_diagram.png" width="250" height="200">
</p>



	
## Development Plans
#### DONE

- [x] Used spike distance and genetic algorithms to optimize network spike raster activity.
- [x] Used pythons NetworkUnit to validate results and t-tests of results
- [x] Created single cell model fitting to Allen Brain Observatory Spike Train Data.
- [x] Ability to toggle between simulator backends (https://github.com/AStupidBear/SpikingNeuralNetworks.jl vs https://github.com/darsnack/SpikingNN.jl)

#### TODO
- [ ] Use large SNNs to optimize smaller SNNs themselves, as this would be parsimonious.
- [ ] Implemented multi-processing of feature extraction/spike distance (sort of)
- [ ] Animation of Genetic Algorithm Convergence (sort of metaheuristics does this with minimal effort)
- [ ] ADAM-Opt predictions using evolved population see file mwe.jl.
- [ ] Read in and optimize against FPGA Event Stream Data AEDAT


#### Why Not Optimize Small SNNs With Bigger SNNs?
This is the long term intended approach, to use a recurrent Inhibitory population + Excitatory population
Network to optimize smaller networks.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://russelljjarvis.github.io/home/"><img src="https://avatars.githubusercontent.com/u/7786645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Russell Jarvis</b></sub></a><br /><a href="https://github.com/russelljjarvis/SpikeNetOpt.jl/commits?author=russelljjarvis" title="Code">💻</a> <a href="https://github.com/russelljjarvis/SpikeNetOpt.jl/commits?author=russelljjarvis" title="Documentation">📖</a> <a href="#ideas-russelljjarvis" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-russelljjarvis" title="Design">🎨</a> <a href="#infra-russelljjarvis" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/mohitsaxenaknoldus"><img src="https://avatars.githubusercontent.com/u/76725454?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mohit Saxena</b></sub></a><br /><a href="https://github.com/russelljjarvis/SpikeNetOpt.jl/commits?author=mohitsaxenaknoldus" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/PallHaraldsson"><img src="https://avatars.githubusercontent.com/u/8005416?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Páll Haraldsson</b></sub></a><br /><a href="https://github.com/russelljjarvis/SpikeNetOpt.jl/commits?author=PallHaraldsson" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

