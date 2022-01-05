using UnicodePlots
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
using SpikeSynchrony
using Statistics
using JLD
using Plots
using UnicodePlots
using Evolutionary
using Random
#using SparseArrays
using Revise
##

# Override to function to include a state.
##
SNN.@load_units
unicodeplots()


const Ne = 200;
const Ni = 50
const seed = 10
const k = 0.5
const this_size = 50


const Ne = 200;
const Ni = 50
const σee = 1.0
const pee = 0.5
const σei = 1.0
const pei = 0.5

global E
global spkd_ground



function make_net_from_graph_structure(Int::arbitrary_int)
    """
    # Circular ladder simulation
    Input integers affect the number of network nodes (neurons) in the circular ladder network.
    """

    arbitrary_int = Int(round(arbitrary_int))
    h = circular_ladder_graph(arbitrary_int)
    hi = circular_ladder_graph(arbitrary_int)
    E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = 1.0)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 1.0)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 1.0)
    P = [E, I]
    C = [EI, IE, II]
    EE = SNN.SpikingSynapse(E, E, :v; σ = 0.5, p = 0.8)
    @inbounds for (i, j) in enumerate(h.fadjlist)
        @inbounds for k in j
            SNN.connect!(EE, i, k, 10)
        end
    end
    @inbounds for (i, j) in enumerate(hi.fadjlist)
        @inbounds for k in j
            if i < Ni && k < Ni
                SNN.connect!(EI, i, k, 10)
                SNN.connect!(IE, i, k, 10)
                SNN.connect!(II, i, k, 10)
            end
        end
    end

    P = [E, I]
    C = [EE, EI, IE, II]
    return P, C

end


function make_net_SNN(Ne, Ni; σee = 1.0, pee = 0.5, σei = 1.0, pei = 0.5)
    Ne = 200
    Ni = 50

    E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
    EE = SNN.SpikingSynapse(E, E, :v; σ = σee, p = pee)
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = pei)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.5)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.5)
    P = [E, I]
    C = [EE, EI, IE, II]
    @show(C)

    return P, C
end
function get_trains(p)
    fire = p.records[:fire]
    x, y = Float32[], Float32[]
    for time in eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(x, time)
            push!(y, neuron_id)
        end
    end
    cellsa = Array{Union{Missing,Any}}(undef, 1, Int(findmax(y)[1]))
    nac = Int(findmax(y)[1])
    for (inx, cell_id) in enumerate(1:nac)
        cellsa[inx] = []
    end
    @inbounds for cell_id in unique(y)
        @inbounds for (time, cell) in collect(zip(x, y))
            if Int(cell_id) == cell
                append!(cellsa[Int(cell_id)], time)

            end

        end
    end

    cellsa

end


function rmse(spkd)
    error = Losses(mean(spkd), spkd; agg = mean)
end

function rmse_depr(spkd)
    total = 0.0
    @inbounds for i = 1:size(spkd, 1)
        total += (spkd[i] - mean(spkd[i]))^2.0
    end
    return sqrt(total / size(spkd, 1))
end


function raster_difference(spkd0, spkd_found)
    maxi0 = size(spkd0)[2]
    maxi1 = size(spkd_found)[2]
    mini = findmin([maxi0, maxi1])[1]
    spkd = ones(mini)
    maxi = findmax([maxi0, maxi1])[1]

    if maxi > 0
        if maxi0 != maxi1
            return sum(ones(maxi))

        end
        if isempty(spkd_found[1, :])
            return sum(ones(maxi))
        end
    end
    spkd = ones(mini)
    @inbounds for i in eachindex(spkd)
        if !isempty(spkd0[i]) && !isempty(spkd_found[i])
            maxt1 = findmax(spkd0[i])[1]
            maxt2 = findmax(spkd_found[i])[1]
            maxt = findmax([maxt1, maxt2])[1]
            if maxt1 > 0.0 && maxt2 > 0.0
                t, S = SpikeSynchrony.SPIKE_distance_profile(
                    unique(sort(spkd0[i])),
                    unique(sort(spkd_found[i]));
                    t0 = 0.0,
                    tf = maxt,
                )
                spkd[i] = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1])
            end
        end
    end
    spkd
end

function loss(model)
    @show(Ne, Ni)
    @show(model)

    σee = model[1]
    pee = model[2]
    σei = model[3]
    pei = model[4]
    P1, C1 = make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    @show(C1)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")
    SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground, spkd_found)
    error = sum(error)
    @show(error)

    error

end


function eval_best(params)
    """
    Evaluate the best candidate network model chosen by a genetic algorithm.
    """

    σee = params[1]
    pee = params[2]
    σei = params[3]
    pei = params[4]
    P1, C1 = make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth: \n")
    SNN.raster([E]) |> display
    println("candidate: \n")
    SNN.raster([E1]) |> display
    E1, spkd_found

end


function Evolutionary.trace!(
    record::Dict{String,Any},
    objfun,
    state,
    population,
    method::GA,
    options,
)
    idx = sortperm(state.fitpop)
    record["fitpop"] = state.fitpop[:]
    record["pop"] = population[:]
end
