module SpikeNetOpt

using PyCall
using OrderedCollections
using LinearAlgebra
using SpikeSynchrony
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
SNN.@load_units
using Evolutionary
import DataStructures
using JLD
using UnicodePlots
using Statistics
using JLD
using Distributed
using Plots
using Evolutionary
using Distributions
using LightGraphs
using Metaheuristics
using SignalAnalysis

include("utils.jl")
include("sdo_network.jl")

end
