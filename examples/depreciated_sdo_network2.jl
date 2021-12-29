using UnicodePlots
import Pkg
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
using SpikeSynchrony
using Statistics
using JLD
using Distributed
using SharedArrays
using Plots
using UnicodePlots
using Evolutionary
using Distributions
using LightGraphs
using Metaheuristics

##
# Override to function to include a state.
##
SNN.@load_units
unicodeplots()



    #outputs = simulate!(pop, T; cb = cb, inputs=input)

global Ne = 200;
global Ni = 50
global σee = 1.0
global pee = 0.5
global σei = 1.0
global pei = 0.5

function make_net_from_graph_structure(xx)#;

    xx = Int(round(xx))
    #@show(xx)
    #h = turan_graph(xx, xx)#, seed=1,cutoff=0.3)

    h = circular_ladder_graph(xx)#, xx)#, seed=1,cutoff=0.3)
    #hi = circular_ladder_graph(xx)#, seed=1,cutoff=0.3)
    E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
    #EE = SNN.SpikingSynapse(E, E, :v; σ = σee, p = 1.0)
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = 1.0)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 1.0)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 1.0)
    # PINningSynapse
    P = [E, I]#, EEA]
    C = [EI, IE, II]#, EEA]
    #EE = SNN.PINningSynapse(E, E, :v; σ=0.5, p=0.8)
    #for n in 1:(N - 1)
    #    SNN.connect!(EE, n, n + 1, 50)
    #end
    #for (i,j) in enumerate(h.fadjlist) println(i,j) end
    EE = SNN.SpikingSynapse(E, E, :v; σ=0.5, p=0.8)

    @inbounds for (i,j) in enumerate(h.fadjlist)
        @inbounds for k in j
            SNN.connect!(EE,i, k, 10)
        end
    end

    @inbounds for (i,j) in enumerate(hi.fadjlist)
        @inbounds for k in j
            if i<Ni && k<Ni

                SNN.connect!(EI,i, k, 10)
                SNN.connect!(IE,i, k, 10)
                SNN.connect!(II,i, k, 10)
            end
        end
    end

    #for (i,j) in enumerate(h.fadjlist)
    #    for k in j
    #        SNN.connect!(EI, i, k, 50)
    #    end
    #end
    P = [E, I]#, EEA]
    C = [EE, EI, IE, II]#, EEA]
    return P, C

end


function make_net(Ne, Ni; σee = 1.0, pee = 0.5, σei = 1.0, pei = 0.5)
     Ne = 200;
     Ni = 50

     E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
     I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
     EE = SNN.SpikingSynapse(E, E, :v; σ = σee, p = pee)
     EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = pei)
     IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.5)
     II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.5)
     P = [E, I]#, EEA]
     C = [EE, EI, IE, II]#, EEA]
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
global E
global spkd_ground

#P, C = make_net(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8, a = 0.02)
#sggcu =[ CuArray(convert(Array{Float32,1},sg)) for sg in spkd_ground ]

#Flux.SGD
#Flux.gpu

function rmse(spkd)
    error = Losses(mean(spkd),spkd;agg=mean)
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

#=
using Base.Threads

function raster_difference_threads(spkd0, spkd_found)
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
    N = length(spkd)
    Threads.foreach(
        Channel(nthreads()) do chnl
            for i in 1:N
                val = 1.0
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
                        val = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1])
                    end
                end
                put!(chnl, val)
                println("thread ", threadid(), ": ", "put ", val)
            end
        end
    end
    spkd
end
=#


function loss(model)
    #σee = model[1]
    #pee = model[2]
    #σei = model[3]
    #pei = model[4]
    #P1, C1 = make_net(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    #@show(model)
    println("best candidate ",26)
    println(" ")
    #println("found ", model[1])
    P1, C1 = make_net_SNN(model[1])#,a=a)

    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    #println("Ground Truth \n")
    #SNN.raster([E]) |> display
    #println("Best Candidate \n")

    #SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground, spkd_found)
    error = sum(error)
    #@show(error)
    #println("Broke here")

    error

end



function eval_best(params)
    xx = Int(round(params[1]))
    @show(xx)
    P1, C1 = make_net_SNN(xx)#,a=a)
    println("found ",xx)

    #σee = params[1]
    #pee = params[2]
    #σei = params[3]
    #pei = params[4]
    #P1, C1 = make_net(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth: \n")
    SNN.raster([E]) |> display
    println("candidate: \n")

    SNN.raster([E1]) |> display
    #error = raster_difference(spkd_ground,spkd_found)
    E1, spkd_found

end


function init_b(lower, upper)
    gene = []

    for (i, (l, u)) in enumerate(zip(lower, upper))
        p1 = rand(l:u, 1)
        append!(gene, p1)
    end
    gene
end

function initf(n)
    genesb = []
    for i = 1:n
        genes = init_b(lower, upper)
        append!(genesb, [genes])
    end
    genesb
end



function initd()
    population = initf(10)
    garray = zeros((length(population)[1], length(population[1])))
    for (i, p) in enumerate(population)
        garray[i, :] = p
    end
    garray[1, :]
end
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
    idx = sortperm(state.fitpop)
    record["fitpop"] = state.fitpop[:]#idx[1:last(idx)]]
    record["pop"] = population[:]
    #record["σ"] = state.
end


#lower = Float32[0.0 0.0 0.0 0.0]# 0.03 4.0]
#upper = Float32[1.0 1.0 1.0 1.0]# 0.2 20.0]

#lower = Int32[3]# 0.0 0.0 0.0]# 0.03 4.0]
#upper = Int32[40]# 1.0 1.0 1.0]# 0.2 20.0]

#lower = Float32[0.0 0.0 0.0 0.0]# 0.03 4.0]
#upper = Float32[1.0 1.0 1.0 1.0]# 0.2 20.0]
#lower = vec(lower)
#upper = vec(upper)
#using Evolutionary, MultivariateStats
#range = Any[lower,upper]

# mutation = domainrange(fill(0.5,4))
#
#function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
#    idx = sortperm(state.fitpop)
#record["population"] = population
#    state.fitpop[idx[1:5]]
#    record["fitpop"] = state.fitpop[idx[1:5]]
#end
#using Evolutionary

#=
function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
    @show(typeof(fitness))
    fitness = SharedArrays.SharedArray{Float32}(fitness)
    @time @sync @distributed for i in 1:length(population)

    #pmap(f, [::AbstractWorkerPool], c...; distributed=true, batch_size=) -> collection
        fitness[i] = value(objfun, population[i])
        #println("I'm worker $(myid()), working on i=$i")
    end
    @show(typeof(fitness))

    fitness
    #fitness = Array(fitness)
    #@show(fitness)
end
=#
