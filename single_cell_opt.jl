ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Pkg
using PyCall
using OrderedCollections
using LinearAlgebra
#using NSGAII
using SpikeSynchrony
using SpikingNeuralNetworks
#include("../src/SpikingNeuralNetworks.jl")
#include("../src/unit.jl")
SNN = SpikingNeuralNetworks
SNN.@load_units

using Evolutionary, Test, Random
#using Plots
import DataStructures
using JLD
using Evolutionary, Test, Random
#unicodeplots()
global ngt_spikes
global opt_vec
global extremum
global extremum_param
global ngt_spikes
global fitness

SNN = SpikingNeuralNetworks
adparam = SNN.ADEXParameter(;a = 6.050246708405076, b = 7.308480222357973,
    cm = 803.1019662706587,
    v_rest= -63.22881649139353,
    tau_m=19.73777028610565,
    tau_w=351.0551915202058,
    v_thresh=-39.232165554444265,
    delta_T=6.37124632135508,
    v_reset = -59.18792270568965,
    spike_height = 16.33506432689027)

E = SNN.AD(;N = 1, param=adparam)
E.I = [795.57128906]
#{'value': array(460.57128906) * pA}
# -

SNN.monitor(E, [:v,:I])


function SNN.sim!(P, C; dt = 0.25ms, simulation_duration = 1300ms, delay = 300ms,stimulus_duration=1000ms)
    temp = deepcopy(P[1].I)
    size = simulation_duration/dt
    cnt1 = 0
	if hasproperty(P[1], :spike_raster )
		P[1].spike_raster::Vector{Int32} = zeros(trunc(Int, size))

	end
    for t = 0ms:dt:simulation_duration
        cnt1+=1
        if cnt1 < delay/dt
           P[1].I[1] = 0.0
        end
        if cnt1 > (delay/dt + stimulus_duration/dt)
	       P[1].I[1] = 0.0
        end
        if (delay/dt) < cnt1 < (stimulus_duration/dt)
           P[1].I[1] = maximum(temp[1])
        end
        SNN.sim!(P, C, dt)
    end
end
#SNN.sim! = sim!
#SNN.sim!([Etest],[0],dt = 0.25ms, simulation_duration = 2000ms, delay = 500ms,stimulus_duration=2000ms)
SNN.sim!([E],[0],dt = 1ms, simulation_duration = 3000ms, delay = 500ms,stimulus_duration=2000ms)

#SNN.vecplot(E, :v) |> display


if false
    isfile("ground_truth.jld")
    vmgtv = load("ground_truth.jld","vmgtv")
    ngt_spikes = load("ground_truth.jld","ngt_spikes")
    gt_spikes = load("ground_truth.jld","gt_spikes")

    ground_spikes = gt_spikes
    ngt_spikes = size(gt_spikes)[1]
    vmgtt = load("ground_truth.jld","vmgtt")

    #plot(Plot(vmgtv,vmgtt,w=1))

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
println("gets here a")

#plot(vmgtt[:],vmgtv[:]) |> display
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms
ranges_adexp = DataStructures.OrderedDict{String,Tuple{Float32,Float32}}()
println("gets here b")

ranges_adexp[:"a"] = (2.0, 20)
ranges_adexp[:"b"] = (2.0, 20)
ranges_adexp[:"cm"] = (12.0, 983.5)
ranges_adexp[:"v_rest"] = (-80, -45)
ranges_adexp[:"tau_m"] = (10.0, 62.78345)
ranges_adexp[:"tau_w"] = (50.0, 354.0)  # Tau_w 0, means very low adaption
ranges_adexp[:"v_thresh"] = (-36.0, -15.0)
ranges_adexp[:"delta_T"] = (1.0, 10.0)
ranges_adexp[:"v_reset"] = (-70.0, -15.0)

ranges_adexp[:"spike_height"] = (1.25, 20.0)

ranges_izhi = DataStructures.OrderedDict{Char,Float32}()
ranges_izhi = ("a"=>(0.002,0.3),"b"=>(0.02,0.36),"c"=>(-75,-35),"d"=>(0.005,16))#,"I"=>[100,9000])

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

function raster_synchp(p)
    fire = p.records[:fire]
    spikes = Float32[]
    #neurons = Float32[]#, Float32[]
    for time = eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(spikes,time)
        end
    end
    spikes
end
function loss(E,ngt_spikes,ground_spikes)
    spikes = raster_synchp(E)
    spikes = [s/1000.0 for s in spikes]
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

function test_current(param,current,ngs)
    #println(param)
    if 1==2
        CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
        E = SNN.IZ(;N = 1, param = CH)
        SNN.monitor(E, [:v])
        SNN.monitor(E, [:fire])
    end


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


    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])

    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms

    E.I = [current*nA]#[param[5]*nA]
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)

    spikes = raster_synchp(E)
    spikes = [s/1000.0 for s in spikes]
    nspk = size(spikes)[1]
    delta = abs(nspk - ngs)
    return nspk
end
function test_c(check_values,param,ngt_spikes,current_dict)
    nspk = -1.0
    for i_c in check_values
        nspk = test_current(param,i_c,ngt_spikes)
        current_dict[nspk] = i_c
        if nspk == ngt_spikes
            if ngt_spikes in keys(current_dict)
                return current_dict,nspk
            end
        end
    end
    return current_dict,nspk
end

function findnearest(x::Array{Any,1}, val::Int64)
    ibest = first(eachindex(x))#[1]
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            dxbest = dx
            ibest = I
        end
    end
    x[ibest]
end

function current_search(param,ngt_spikes)

    current_dict = Dict()
    minc = 0.00005
    maxc = 99999.0
    step_size = (maxc-minc)/10.0
    check_values = minc:step_size:maxc
    current_dict = Dict()
    cnt = 0.0
    while (ngt_spikes in keys(current_dict))==false
        current_dict,nspk = test_c(check_values,param,ngt_spikes,current_dict)
        over_s = Dict([(k,v) for (k,v) in current_dict if k>ngt_spikes])
        under_s = Dict([(k,v) for (k,v) in current_dict if k<ngt_spikes])
        if length(over_s)>0
            # find the lowest current that caused more than one spike
            # throw away part of dictionary that has no spikes
            # find minimum of value in the remaining dictionary
            new_top = findmin(collect(values(over_s)))[1]

        else
            new_top = maxc*10

        end
        if length(under_s)>0
            new_bottom = findmax(collect(values(under_s)))[1]

        else
            new_bottom = minc-minc*(1.0/2.0)
        end

        step_size = abs(new_top-new_bottom)/10.0
        if step_size==0

            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            #println("spike disparity $closest")
            return closest

        end
        check_values = new_bottom:step_size:new_top
        #println(new_bottom,new_top)

        cnt+=1

        if cnt >150
            #println("intended number spikes $ngt_spikes")

            #println(current_dict)
            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            #println("spike disparity $closest")
            return closest
            #findmin(collect(values(current_dict)))[1]
        end
    end
    return current_dict[ngt_spikes]
end


function zz_(param)

    ###
    #CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    #E = SNN.IZ(;N = 1, param = CH)
    ###
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


    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    #T = (ALLEN_DURATION+ALLEN_DELAY)*ms
    current = current_search(param,ngt_spikes)
    #println(current)
    E.I = [current*nA]#[param[5]*nA]

    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    error = loss(E,ngt_spikes,ground_spikes)
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
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
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
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    #vec = SNN.vecplot(E, :v)
    #vec |> display
    #vec


end


function initd()
    population = initf(10)
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
