module SpikeNetOpt

    using PyCall
    using OrderedCollections
    using LinearAlgebra
    using SpikeSynchrony
    using SpikingNeuralNetworks
    SNN = SpikingNeuralNetworks
    SNN.@load_units
    using Evolutionary#, Test, Random
    import DataStructures
    using JLD
    #using Reexport
    using UnicodePlots
    using Statistics
    using JLD
    using Distributed
    using SharedArrays
    using Plots
    using Evolutionary
    using Distributions
    using LightGraphs
    using Metaheuristics
    using ClearStacktrace
    using SharedArrays
    using WaspNet
    using SpikingNN
    using SignalAnalysis
    #using Reexport
    #using Spikes
    # Export Function names and struct names found in file utils
    export get_trains
    export get_vm
    function get_trains end
    function get_vm end
    function checkmodel end
    function make_net_SNN end
    include("utils.jl")
    include("current_search.jl")
    #include("spike_distance_opt.jl")
    include("sdo_network.jl")

    @show(make_net_SNN)
    #include("sciunit.jl")

end
