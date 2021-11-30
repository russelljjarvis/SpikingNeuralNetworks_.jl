



using Pkg
Pkg.add("PyPlot")
Pkg.add(url = "https://github.com/russelljjarvis/SpikeSynchrony.jl")
Pkg.add(url="https://github.com/paulmthompson/SpikeSorting.jl.git")
Pkg.add(url="https://github.com/paulmthompson/Spikes.jl.git")
Pkg.add("Conda")
Pkg.add("PyCall")
run(`$(which pip) install -networkunit`)

#Pkg.add("SharedArrays")
#Pkg.add("Requires")
#Pkg.add("UnPack")
#Pkg.add("LightGraphs")
#Pkg.add("Flux")
#Pkg.add("PyPlot")
#Pkg.add("OrderedCollections")
#Pkg.add("Evolutionary")
#Pkg.add("JLD")
#Pkg.add("DataStructures")
#Pkg.add("Unitful")
