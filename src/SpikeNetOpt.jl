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

using Reexport

using UnicodePlots
#using Evolutionary, Test, Random
#using Pkg

#import Pkg
#using SpikingNeuralNetworks
#SNN = SpikingNeuralNetworks
using SpikeSynchrony
using Statistics
using JLD
using Distributed
using SharedArrays
using Plots
using UnicodePlots
#using Evolutionary
using Distributions
using LightGraphs
using Metaheuristics


# Export Function names and struct names found in file utils
export get_vm
function get_vm end
function checkmodel end
include("utils.jl")
include("current_search.jl")
#include("sciunit.jl")




end
