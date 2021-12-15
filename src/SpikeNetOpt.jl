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
    #using WaspNet
    #using SpikingNN
    using SignalAnalysis
    #using Reexport
    #using Spikes
    # Export Function names and struct names found in file utils
    #=
    export get_trains
    export get_vm
    export sim_net_darsnack

    function get_trains end
    function get_vm end
    function checkmodel end
    function make_net_from_graph_structure end
    function make_net end
    function spike_train_difference end
    function get_spikes end
    function get_trains end
    function sim_net_darsnack end
    =#
    include("utils.jl")
    #include("current_search.jl")
    include("sdo_network.jl")

    #include("sciunit.jl")

end
