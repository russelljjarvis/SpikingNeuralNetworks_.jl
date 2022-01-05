# Optimize a network derived from a cicular ladder shaped connectome.

using Plots
using SpikeNetOpt
SNO = SpikeNetOpt
using SpikingNeuralNetworks
using Evolutionary
SNN = SpikingNeuralNetworks
using Metaheuristics
#using DrWatson
#using Pkg
SNN.@load_units


##
# Ground truths
##
const Ne = 200;
const Ni = 50
const σee = 1.0
const pee = 0.5
const σei = 1.0
const pei = 0.5
MU = 10


const E
const spkd_ground
const GT = 26

g, Cg = SpikeNetOpt.make_net_from_graph_structure(GT)
P, C = SpikeNetOpt.make_net_SNN(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8)
E, I = P
EE, EI, IE, II = C
SNN.monitor([E, I], [:fire])
sim_length = 1000
@inbounds for t = 1:sim_length*ms
    E.I = vec([11.5 for i = 1:sim_length])
    SNN.sim!(P, C, 1ms)

end
spkd_ground = SpikeNetOpt.get_trains(P[1])
display(SNN.raster(P[1]))
SNN.raster(P[1]) |> display
META_HEUR_OPT = true

function loss(model)
    """
    A loss function for calculating errors in machine learning.
    """
    σee = model[1]
    pee = model[2]
    σei = model[3]
    pei = model[4]
    P1, C1 = SNO.make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 1000
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")
    SNN.raster([E1]) |> display
    error = SNO.spike_train_difference(spkd_ground, spkd_found)
    error = sum(error)
    @show(error)
    error
end

D = 10
bounds = [3ones(D) 40ones(D)]'
a = view(bounds, 1, 1)
b = view(bounds, 1, 2)
information = Information(f_optimum = 0.0)
options = Options( seed = 1, iterations=10, f_calls_limit =10)

D = size(bounds, 2)
nobjectives=1

options = Options( seed = 1, iterations=10000, f_calls_limit = 25000)
npartitions = nobjectives == 2 ? 100 : 12

methods = [
        SMS_EMOA(N = 5, n_samples=5, options=options),
        NSGA2(options=options)
        ]


for method in methods
    f_calls = 0
    result = optimize(SpikeNetOpt.loss, bounds, method)
    @show(result)

end
