
using Test
using SpikingNeuralNetworks
using Plots
SNN = SpikingNeuralNetworks
SNN.@load_units

for tests in ["spike_current_search_test.jl"]
    include(tests)
end
