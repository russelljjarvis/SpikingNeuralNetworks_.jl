ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Pkg
using PyCall
using OrderedCollections
using LinearAlgebra
using SpikeSynchrony
using SpikingNeuralNetworks
include("current_search.jl")
SNN = SpikingNeuralNetworks
SNN.@load_units

using Evolutionary, Test, Random
import DataStructures
using JLD
using Evolutionary, Test, Random

global ngt_spikes
global opt_vec
global extremum
global extremum_param
global ngt_spikes
global fitness

using Plots
unicodeplots()
function vecplot(p, sym)
    v = SNN.getrecord(p, sym)
    y = hcat(v...)'
    x = 1:length(v)
    plot(x, y, leg = :none,
    xaxis=("t", extrema(x)),
    yaxis=(string(sym), extrema(y)))
end

function vecplot(P::Array, sym)
    plts = [vecplot(p, sym) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

#SNN.sim! = sim!
if isfile("ground_truth.jld")
    vmgtv = load("ground_truth.jld","vmgtv")
    ngt_spikes = load("ground_truth.jld","ngt_spikes")
    gt_spikes = load("ground_truth.jld","gt_spikes")

    ground_spikes = gt_spikes
    ngt_spikes = size(gt_spikes)[1]
    vmgtt = load("ground_truth.jld","vmgtt")
    plot(plot(vmgtv,vmgtt,w=1))

else
    py"""
    from neo import AnalogSignal
    from neuronunit.allenapi import make_allen_tests_from_id

    """

    py"""
    specimen_id = (
        325479788,
        324257146,
        476053392,
        623893177,
        623960880,
        482493761,
        471819401
    )
    specimen_id = specimen_id[1]
    target_num_spikes=7
    sweep_numbers, data_set, sweeps = make_allen_tests_from_id.allen_id_to_sweeps(specimen_id)
    (vmm,stimulus,sn,spike_times) = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(
        target_num_spikes, data_set, sweep_numbers, specimen_id
    )
    """
    gt_spikes = py"spike_times"
    ground_spikes = gt_spikes

    ngt_spikes = size(gt_spikes)[1]
    vmgtv = py"vmm.magnitude"
    vmgtt = py"vmm.times"

    save("ground_truth.jld", "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)
    filename = string("ground_truth: ", py"target_num_spikes")#,py"specimen_id)
    filename = string(filename,py"specimen_id")
    filename = string(filename,".jld")
    save(filename, "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)

end

#plot(vmgtt[:],vmgtv[:]) |> display
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms
ranges_adexp = DataStructures.OrderedDict{String,Tuple{Float32,Float32}}()


#@show(E)

#SNN = SpikingNeuralNetworks
adparam = SNN.ADEXParameter(;a = 6.050246708405076, b = 7.308480222357973,
    cm = 803.1019662706587,
    v0= -63.22881649139353,
    τ_m=19.73777028610565,
    τ_w=351.0551915202058,
    θ=-39.232165554444265,
    delta_T=6.37124632135508,
    v_reset = -59.18792270568965,
    spike_delta = 16.33506432689027)

ranges_adexp[:"a"] = (2.0, 10)
ranges_adexp[:"b"] = (5.0, 10)
ranges_adexp[:"cm"] = (700.0, 983.5)
ranges_adexp[:"v0"] = (-70, -55)

ranges_adexp[:"τ_m"] = (10.0, 42.78345)
ranges_adexp[:"τ_w"] = (300.0, 454.0)  # Tau_w 0, means very low adaption
ranges_adexp[:"θ"] = (-45.0,-10)
ranges_adexp[:"delta_T"] = (1.0, 5.0)
ranges_adexp[:"v_reset"] = (-70.0, -15.0)
ranges_adexp[:"spike_delta"] = (1.25, 20.0)

ranges_izhi = DataStructures.OrderedDict{Char,Float32}()
ranges_izhi = ("a"=>(0.002,0.3),"b"=>(0.02,0.36),"c"=>(-75,-35),"d"=>(0.005,16))#,"I"=>[100,9000])

izparam = SNN.IZParameter(;a = 0.050246708405076, b = 0.308480222357973, c=-55,d=1.0)
E = SNN.IZ(;N = 1, param=izparam)
#E = SNN.AD(;N = 1, param=adparam)
E.I = [0.979557128906]


#{'value': array(460.57128906) * pA}
# -
SNN.monitor(E, [:v,:I])

SNN.sim!([E],dt = 0.25ms, simulation_duration = 2000ms, delay = 500ms,stimulus_duration=2000ms)

v = vecplot(E, :v)
v |> display
#v = vecplot(E, :v)
@show(v)


E = SNN.AD(;N = 1, param=adparam)
E.I = [5995.57128906]



SNN.monitor(E, [:v,:I])
SNN.sim!([E],dt = 1ms, simulation_duration = 3000ms, delay = 500ms,stimulus_duration=2000ms)

vecplot(E, :v) |> display
v = vecplot(E, :v)
@show(v)


function get_ranges(ranges)

    lower = []
    upper = []
    for (k,v) in ranges
        append!(lower,v[1])
        append!(upper,v[2])
    end
    lower,upper
end

function init_b(lower,upper)
    gene = []
    #chrome = Float32[size(lower)[1]]
    for (i,(l,u)) in enumerate(zip(lower,upper))
        p1 = rand(l:u, 1)
        append!(gene,p1)
        #chrome[i] = p1
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
#lower,upper = get_ranges()

lower,upper = get_ranges(ranges_adexp)


function loss(E,ngt_spikes,ground_spikes)
    spikes = raster_synchp(E)
    spikes = [s*ms for s in spikes]
    maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkd = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkd = 1.0
    end
    delta = abs(size(spikes)[1] - ngt_spikes)
    return spkd+delta

end



function zz_(param)

    ###
    #CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    #E = SNN.IZ(;N = 1, param = CH)
    ###

    adparam = SNN.ADEXParameter(;a = param[1],
        b = param[2],
        cm = param[3],
        v0 = param[4],
        τ_m = param[5],
        τ_w = param[6],
        θ = param[7],
        delta_T = param[8],
        v_reset = param[9],
        spike_delta = param[10])

    E = SNN.AD(;N = 1, param=adparam)


    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    #T = (ALLEN_DURATION+ALLEN_DELAY)*ms
    current = current_search(param,ngt_spikes)
    #println(current)
    E.I = [current*nA]#[param[5]*nA]

    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    vecplot(E, :v) |> display

    error = loss(E,ngt_spikes,ground_spikes)
    @show(error)
    error
end


function z(x::AbstractVector)
    param = x
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [param[5]*nA]
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    error = loss(E,ngt_spikes)
    error
end


function checkmodel(param)

    adparam = SNN.ADEXParameter(;a = param[1],
        b = param[2],
        cm = param[3],
        v_rest = param[4],
        tau_m = param[5],
        tau_w = param[6],
        v_thresh = param[7],
        delta_T = param[8],
        v_reset = param[9],
        spike_height = param[10])

    E = SNN.AD(;N = 1, param=adparam)


    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [current_search(param,ngt_spikes)*nA]

    SNN.monitor(E, [:v])
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    #vec = SNN.vecplot(E, :v)
    #vec |> display
    #vec


end


function initd()
    population = initf(50)
    garray = zeros((length(population)[1], length(population[1])))
    for (i,p) in enumerate(population)
        garray[i,:] = p
    end
    garray[1,:]
end
Evolutionary.ConstraintBounds(lower,upper,lower,upper)
lw = domainrange(lower)
up = domainrange(upper)
dr=[]
for (r0,r1) in values(ranges_adexp)
    dh = domainrange([r0,r1])
end

ɛ = 0.125
options = GA(
    populationSize = 50,
    ɛ = ɛ,
    selection = ranklinear(1.5),#ss,
    crossover = intermediate(1.0),#line(1.0),#xovr,
    mutation = uniform(1.0),#domainrange(fill(1.0,ts)),#ms
)
@time result = Evolutionary.optimize(zz_,lower,upper, initd, options,
    Evolutionary.Options(iterations=15, successive_f_tol=25, show_trace=true, store_trace=true)
)
fitness = minimum(result)
println("GA: ɛ=$ɛ) => F: $(minimum(result))")# C: $(Evolutionary.iterations(result))")
extremum_param = Evolutionary.minimizer(result)
opt_vec = checkmodel(extremum_param)
println("probably jumbled extremum param")
println(extremum_param)
plot(opt_vec)|> display
plot(vmgtt[:],vmgtv[:])|> display
#, label = "randData", ylabel = "Y axis",color = :red, legend = :topleft, grid = :off, xlabel = "Numbers Rand")
#p = twiny()
#plot!(p,vec, label = "log(x)", legend = :topright, box = :on, grid = :off, xlabel = "Log values") |> display
#return fitness
#end
#end
#break
#end
