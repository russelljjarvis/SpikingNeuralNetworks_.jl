
# It is mostly the case that I am simulating the highly specific random data that the genetic algorithm uses to fit another simulated model to. I call this `model2model` data fitting.



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



makes a ground truth model, and finds network parameters for all of the values:
julia
include("run_net_opt.jl")


```
σee = 1.0, pee = 0.5, σei = 1.0, pei = 0.5, a = 0.02)
```
julia

include("evolutionary_based_opt.jl")
