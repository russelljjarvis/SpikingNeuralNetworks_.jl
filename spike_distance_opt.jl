using UnicodePlots
import Pkg
using Flux
include("../src/SpikingNeuralNetworks.jl")
SNN = SpikingNeuralNetworks
using SpikeSynchrony
ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Statistics
using Pkg
using JLD
#using PyCall
using Evolutionary#, Test, Random
using Distributed
using SharedArrays
include("../src/plot.jl")
using Plots
using UnicodePlots
#dir(x) = fieldnames(typeof(x))
using FoldsCUDA, CUDA, FLoops
using GPUArrays: @allowscalar

SNN.@load_units

 ###
# Network 1.
###

Ne = 200;
Ni = 50
function make_net(Ne,Ni; σee= 1.0,  pee= 0.5,σei = 1.0,  pei= 0.5, a=0.02)
    E = SNN.IZ(;N = Ne, param = SNN.IZParameter(;a = a, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(;N = Ni, param = SNN.IZParameter(;a = 0.1, b = 0.2, c = -65, d = 2))
    EE = SNN.SpikingSynapse(E, E, :v; σ = σee,  p = pee)
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei,  p = pei)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.5)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.5)
    P = [E, I]#, EEA]
    C = [EE, EI, IE, II]#, EEA]
    return P,C

end
function get_trains(p)
    fire = p.records[:fire]
    x, y = Float32[], Float32[]
    for time = eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(x, time)
            push!(y, neuron_id)
        end
    end
    cellsa = Array{Union{Missing,Any}}(undef, 1, Int(findmax(y)[1]))

    ##
    # nac == number_of_active_cells
    ##
    nac = Int(findmax(y)[1])
    for (inx,cell_id) in enumerate(1:nac)
        cellsa[inx] = []
    end
    for (inx,cell_id) in enumerate(unique(y))
        for (index, (time,cell)) in enumerate(collect(zip(x,y)))
            if Int(cell_id) == cell
                append!(cellsa[Int(cell_id)],time)

            end

        end
    end

    cellsa

end
#=
function heat_difference(spkd0,spkd1)
    maxi0 = size(spkd0)[1]
    maxi1 = size(spkd1)[1]
    mini = findmin([maxi0,maxi1])[1]
    spkd = SharedArrays.SharedArray{Float32}(mini)
    maxi = findmax([maxi0,maxi1])[1]

    if maxi>0
        if maxi0!=maxi1
            return sum(ones(maxi))

        end
        if isempty(spkd1[1,:])
            return sum(ones(maxi))
        end
    end
    #@sync @distributed
    @inbounds for i in 1:mini
    #@sync @distributed

        spkd[i] = 1.0
        if !isempty(spkd0[i]) && !isempty(spkd1[i])
            maxt1 = findmax(spkd0[i])[1]
            maxt2 = findmax(spkd1[i])[1]
            maxt = findmax([maxt1,maxt2])[1]

            if maxt1>0.0 &&  maxt2>0.0
                t, S = SpikeSynchrony.SPIKE_distance_profile(unique(sort(spkd0[i])),unique(sort(spkd1[i]));t0=0.0,tf = maxt)
                b = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
                spkd[i] =  SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)

            end
        end
    end
end
=#
global E
global spkd_ground

P,C = make_net(Ne,Ni,σee = 0.5,  pee= 0.8,σei = 0.5,  pei= 0.8,a=0.02)
E, I = P #, EEA]
EE, EI, IE, II = C
SNN.monitor([E, I], [:fire])
#global E_stim = []#Vector
sim_length = 1000
@inbounds for t = 1:sim_length
    #tempne = 5randn(Ne)
    #E.I .= 4.5#tempne#4.25#tempne #[ tempne for x in 1:Ne ]
    E.I = vec([11.5 for i in 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)

    #append!(E_stim,E.I)
    SNN.sim!(P, C, 1ms)

end
#_,_,_,spkd_ground = raster_synchp(P[1])
spkd_ground = get_trains(P[1])
sgg =[ convert(Array{Float32,1},sg) for sg in spkd_ground ]
sggcu =[ CuArray(convert(Array{Float32,1},sg)) for sg in spkd_ground ]

#Flux.SGD
#Flux.gpu
function rmse(spkd)
    total = 0.0
    @inbounds for i in 1:size(spkd, 1)
        total += (spkd[i] - mean(spkd[i]))^2.0
    end
    return sqrt(total / size(spkd, 1))
end

function raster_difference(spkd0,spkd1)
    maxi0 = size(spkd0)[2]
    maxi1 = size(spkd1)[2]
    #@show(maxi0)

    mini = findmin([maxi0,maxi1])[1]
    #@show(mini)
    spkd = ones(mini)#SharedArrays.SharedArray{Float32}(mini)
    maxi = findmax([maxi0,maxi1])[1]

    if maxi>0
        if maxi0!=maxi1
            return sum(ones(maxi))

        end
        if isempty(spkd1[1,:])
            return sum(ones(maxi))
        end
    end
    #@sync @distributed
    #@inbounds
    #spkd = CUDA.ones(mini)


    if 1==2#has_cuda_gpu()
        spkd = CuArray(1:mini)
        spkd1_ = CuArray(1:mini)
        spkd0_ = CuArray(1:mini)
        print("gets here")
        spkd1_ = CuArray(spkd1)
        spkd0_ = CuArray(spkd0)
        @floop CUDAEx() for (x, i) in zip(spkd, eachindex(spkd))
            spkd[i] = 1.0
            if !isempty(spkd0[i]) && !isempty(spkd1[i])

                maxt1 = findmax(spkd0[i])[1]
                maxt2 = findmax(spkd1[i])[1]
                maxt = findmax([maxt1,maxt2])[1]

                if maxt1>0.0 &&  maxt2>0.0
                    t, S = SpikeSynchrony.SPIKE_distance_profile(unique(sort(spkd0[i])),unique(sort(spkd1[i]));t0=0.0,tf = maxt)
                    b = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
                    spkd[i] =  SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)

                end
            end
        end
    else
        spkd = ones(mini)
        @floop for (x, i) in zip(spkd, eachindex(spkd))

        #@inbounds for (x, i) in zip(spkd, eachindex(spkd))
            spkd[i] = 1.0
            if !isempty(spkd0[i]) && !isempty(spkd1[i])

                maxt1 = findmax(spkd0[i])[1]
                maxt2 = findmax(spkd1[i])[1]
                maxt = findmax([maxt1,maxt2])[1]

                if maxt1>0.0 &&  maxt2>0.0
                    t, S = SpikeSynchrony.SPIKE_distance_profile(unique(sort(spkd0[i])),unique(sort(spkd1[i]));t0=0.0,tf = maxt)
                    b = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
                    spkd[i] =  SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)

                end
            end
        end
    end

    #Plots.plot(spkd) |>display
    #Plots.heatmap(spkd)|>display
    #Plots.plot([i for i in 1:mini],[spkd[i] for i in 1:mini])|>display
    #unicodeplots()

    #Plots.scatter([spkd[i] for i in 1:mini])|>display

    #unicodeplots()
    #@show(spkd)
    scatter([i for i in 1:mini],spkd)|>display
    #Plots.plot([i for i in 1:mini],fill(mean(spkd)))|>display
    #scatter(spkd)|>display
    #else
    #temp = spkd .- mean(spkd)
    #error = sqrt(sum(temp.^2.0)/mini)
    error = rmse(spkd)+sum(spkd)
end

#error = raster_difference(spkd0,spkd1)
#println(error)

global Ne = 200;
global Ni = 50

function loss(params)
    σee = params[1]
    pee = params[2]
    σei = params[3]
    pei = params[4]
    #a = params[5]
    #I = params[6]
    #nothin = params[5]
    #println(params)

    #make_net(Ne,Ni; σee= 0.9,  pee= 0.8,σei = 0.5,  pei= 0.8)
    ## Ground truth net.
    # should just use results from single simulation.
    #P,C = make_net(Ne,Ni,σee = 0.59,  pee= 0.58,σei = 0.5,  pei= 0.58)


    P1 , C1 = make_net(Ne,Ni,σee = σee,  pee=pee,σei =σei,  pei= pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 1000
    @inbounds for t in 1:sim_length*ms
        E1.I = vec([11.5 for i in 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd1 = get_trains(P1[1])
    println("GT \n")
    SNN.raster([E]) |> display
    println("candidate \n")

    SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground,spkd1)
    #println(error," the error")
    #@show(error)
    error

end


function eval_best(params)
    σee = params[1]
    pee = params[2]
    σei = params[3]
    pei = params[4]
    P1 , C1 = make_net(Ne,Ni,σee = σee,  pee=pee,σei =σei,  pei= pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 1000
    @inbounds for t in 1:sim_length
        E1.I = vec([11.5 for i in 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd1 = get_trains(P1[1])
    println("GT \n")
    SNN.raster([E]) |> display
    println("candidate \n")

    SNN.raster([E1]) |> display
    #error = raster_difference(spkd_ground,spkd1)
    E1,spkd1

end

#@pyimport networkunit
#py"""


function init_b(lower,upper)
    gene = []

    for (i,(l,u)) in enumerate(zip(lower,upper))
        p1 = rand(l:u, 1)
        append!(gene,p1)
    end
    gene
end

function initf(n)
    genesb = []
    for i in 1:n
        genes = init_b(lower,upper)
        append!(genesb,[genes])
    end
    genesb
end



function initd()
    population = initf(10)
    garray = zeros((length(population)[1], length(population[1])))
    for (i,p) in enumerate(population)
        garray[i,:] = p
    end
    garray[1,:]
end

lower = [0.0 0.0 0.0 0.0]# 0.03 4.0]
upper = [1.0 1.0 1.0 1.0]# 0.2 20.0]
lower = vec(lower)
upper = vec(upper)

#N = size(lower)[1]
#var= domainrange(fill(0.25,N))
#@show(var)
#N = 3
#cmaes = CMAES(;mu=20, lambda=100)
#ga = GA(populationSize=100, ɛ=0.1, selection=rouletteinv, crossover=intermediate(0.25), mutation=domainrange(fill(0.5,N)))
#es = ES(initStrategy=IsotropicStrategy(N), recombination=average, srecombination=average, mutation=gaussian, smutation=gaussian, μ=10, ρ=3, λ=100, selection=:plus)
#de = DE(populationSize = 100)

ɛ = 0.125
options = GA(
    populationSize = 10,
    ɛ = ɛ,
    selection = ranklinear(1.5),#ranklinear(1.5),#ss,
    crossover = intermediate(1.0),#xovr,
    mutation = uniform(1.0),#(.015),#domainrange(fill(1.0,ts)),#ms
)
#println(betweenness_centrality)
#genes = initd()#|>genes
#println(genes)
#initd()|>println()
###
# initd not yet defined
###
result = Evolutionary.optimize(loss,lower,upper, initd, options,
    Evolutionary.Options(iterations=100, successive_f_tol=25, show_trace=true, store_trace=true)
)
fitness = minimum(result)

filename = string("GAsolution.jld")#, py"target_num_spikes")#,py"specimen_id)
params = result.minimizer
E1,spkd1 = eval_best(params)
save(filename,"spkd_ground",spkd_ground,"spkd1",spkd1,"Ne" = Ne,"Ni" = Ni,"sim_length",sim_length)


#println("σee = 0.59,  pee= 0.58,σei = 0.5,  pei= 0.58")
println("best result")
loss(result.minimizer)
println("σee = 0.5,  pee= 0.8,σei = 0.5,  pei= 0.8")

#println("σee = 0.45,  pee= 0.8,σei = 0.4,  pei= 0.9)")
@show(result.minimizer)

@show(fitness)

@show(result)
#end
#dir(result)
run(`python-jl validate_candidate.py`)


#, c = :greys)


#PyPlot.gray()
#imshow(spkd,interpolation="none")
#colorbar()

#imshow(spkd)
#=

using BenchmarkTools
using CuArrays
using GPUifyLoops

@inline incmod1(a, n) = a == n ? 1 : a+1  # Wrap around indexing for i+1.
δx(f, Nx, i, j, k) = f[incmod1(i, Nx), j, k] - f[i, j, k]  # x-difference operator

function time_stepping_kernel(::Val{Dev}, f, δxf) where Dev
    #@setup Dev

    Nx, Ny, Nz = size(f)
    @loop for i in (1:Nx; threadIdx().x)
        for k in 1:Nz, j in 1:Ny
            δxf[i, j, k] = δx(f, Nx, i, j, k)
        end
    end

    @synchronize
end

time_step!(A::Array, B::Array) = time_stepping_kernel(Val(:CPU), A, B)

function time_step!(A::CuArray, B::CuArray)
    @cuda threads=512 time_stepping_kernel(Val(:GPU), A, B)
end

#=
function synchp(P::Array)
    y0 = Int32[0]
    X = Float32[];
    Y = Float32[]
    for p in P
        x, y = raster_synchp(p)
        append!(X, x)
        append!(Y, y)# .+ sum(y0))
        push!(y0, p.N)
        println(p.N)
    end
    return X,Y
end
=#

#X,Y = synchp(P)

for (i,_) in enumerate(unique(cellsa))
    for (j,_) in enumerate(unique(cellsa))
        if i!=j
            maxt = findmax(sort!(unique(vcat(cellsa[i],cellsa[j]))))[1]
            t, S = SPIKE_distance_profile(cellsa[i], cellsa[j];t0=0,tf = maxt)
            #println(t,S)
        end
    end
end
println(cellsa[1])
=#
