# SpikeNetOpt.jl
A julia Spiking Neural Network Optimizer
The Evolutionary.jl package provides Genetic Algorithms that are used to optimize spiking neural networks

The loss function is constructed by computing Spike Distance between all pairs of neurons
Networks are optimized using pair wise spike-distance metric on each pair of neurons
Pythons NetworkUnit package is used to perform a posthoc evaluation of the optimized network.

See the figure below where local variation and firing rates are compared against every neuron between two model networks.
![firing_rates.png](firing_rates.png)
