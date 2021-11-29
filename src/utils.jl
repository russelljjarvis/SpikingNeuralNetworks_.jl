include("current_search.jl")
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
using Evolutionary, Test, Random
using Distributed
using SharedArrays
using SignalAnalysis


function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
    idx = sortperm(state.fitpop)
    record["fitpop"] = state.fitpop[:]#idx[1:last(idx)]]
    record["pop"] = population[:]
end


###
# Code appropriated from:
# https://github.com/JuliaML/MLPlots.jl/blob/master/src/optional/onlineai.jl
###
type SpikeTrains
    n::Int
    plt
end

function SpikeTrains(n::Integer; kw...)
    plt = plot(n; leg=false, yticks=nothing, kw...)
    SpikeTrains(n, plt)
end

const _halfheight = 0.6
function Base.push!(spiketrains::SpikeTrains, idx::Integer, t::Real)
    append!(spiketrains.plt, idx, Float64[NaN, t, t],
            Float64[NaN, idx + _halfheight, idx - _halfheight])
    spiketrains
end
function Base.push!{T<:Real}(spiketrains::SpikeTrains, ts::AbstractVector{T})
    for i in 1:length(ts)
        push!(spiketrains, i, ts[i])
    end
    spiketrains
end

function get_ranges(ranges)

    lower = Float32[]
    upper = []#Float32[]
    for (k,v) in ranges
        append!(lower,v[1])
        append!(upper,v[2])
    end
    lower,upper
end

function init_b(lower,upper)
    gene = []#Float32
    for (i,(l,u)) in enumerate(zip(lower,upper))
        p1 = rand(l:u, 1)
        append!(gene,p1)
    end
    gene
end

function initf(n)
    genesb = []#Float32[]
    for i in 1:n
        genes = init_b(lower,upper)
        append!(genesb,[genes])
    end
    genesb
end

function custom_raster(P1::Array,P2::Array)
    y0 = Int32[0]
    X = Float32[];
	Y = Float32[]
	N=1
	y=0
	for p in [P1,P2]
		y+=1
		append!(X, P1)
		append!(Y, 1 .+ sum(y0))
    	push!(y0, 1)
		append!(X, P2)
		append!(Y, 1 .+ sum(y0))
	    push!(y0, 1)
		N+=1
    end
    plt = scatter(X, Y,w=50, m = (1, :black), leg = :none, marker=:vline,
                  xaxis=("t", (0, Inf)), yaxis = ("neuron",))
    y0 = y0[2:end-1]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
    return plt
end


function custom_raster2(P1::Array,P2::Array)
	#neuron1=1*zeros(length(P1))
	#neuron2 = [2*i for i in neuron1]
	#@show(P1)
	#@show(P2)
    plt3=plot!(P1, seriestype="vline", label="")
	plt4=plot!(P2, seriestype="vline", label="")
	display(plt3)
	display(plt4)

	#plt3 = vline!(P1)#,P1)#,w=50)
	#display(plt3)
	#plt4 = vline!(P2)
	#display(plt4)
	#plot!(plt3,plt3)#,P2)#,w=50)
    #return plt1
end

function get_vm(p,simulation_duration)
    vm = p.records[:v]
	vm = [i[1] for i in vm]
	vm = signal(vm, length(vm)/simulation_duration)
	vm
end

function checkmodel(param,cell_type,ngt_spikes)
    if cell_type=="IZHI"
        pp = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
        E = SNN.IZ(;N = 1, param = pp)
    end
    if cell_type=="ADEXP"
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
    end


    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
	current = current_search(cell_type,param,ngt_spikes)
    E.I = [current*nA]#_search(param,ngt_spikes)*nA]

    SNN.monitor(E, [:v])
	SNN.monitor(E, [:fire])

	###
	# TODO buid t into neuron models
	# SNN.monitor(E, [:t])
    ###
	simulation_duration= ALLEN_DURATION+ALLEN_DELAY+343ms
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = simulation_duration)
	spikes = get_spikes(E)
    spikes = [s/1000.0 for s in spikes]
	vm = get_vm(E,simulation_duration)
	#vm = E.records[:v]
	#vm = [i[1] for i in vm]

	return (vm,spikes)
end
function Evolutionary.value!(::Val{:multiproc}, fitness, objfun, population::AbstractVector{IT}) where {IT}
	fitness = SharedArrays.SharedArray{Float32}(fitness)
	@time @sync @distributed for i in 1:length(population)
		fitness[i] = value(objfun, population[i])
	end
	#@show(fitness)
	fitness
end
function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
	Threads.@threads for i in 1:length(population)
		fitness[i] = value(objfun, population[i])
	end
end
function get_data()
	#print(pwd())
	file="../JLD/ground_truth.jld"
    if false#isfile(file)
        vmgtv = load(file,"vmgtv")
        ngt_spikes = load(file,"ngt_spikes")
        gt_spikes = load(file,"gt_spikes")

        ground_spikes = gt_spikes
        ngt_spikes = size(gt_spikes)[1]
        vmgtt = load(file,"vmgtt")

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

		#s_a = signal(opt_vec, length(opt_vec)/3.5)
		julia_version_vm = signal(vmgtv, length(vmgtt)/last(vmgtt))


		plot(plot(vmgtv,vmgtt,w=1))

        save("ground_truth.jld", "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes,"julia_version_vm",julia_version_vm)
        #filename = string("ground_truth: ", py"target_num_spikes")#,py"specimen_id)
        #filename = string(filename,py"specimen_id")
        #filename = string(filename,".jld")
        #save(filename, "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)

    end

        vmgtv = load("ground_truth.jld","vmgtv")
        ngt_spikes = load("ground_truth.jld","ngt_spikes")
        ngt_spikes = size(gt_spikes)[1]
        ground_spikes = load("ground_truth.jld","gt_spikes")

        vmgtt = load("ground_truth.jld","vmgtt")
		julia_version_vm = load("ground_truth.jld","julia_version_vm")
    return (vmgtv,vmgtt,ngt_spikes,ground_spikes,julia_version_vm)
end
function get_izhi_ranges()
    ranges_izhi = DataStructures.OrderedDict{Char,Float32}()
    ranges_izhi = ("a"=>(0.002,0.3),"b"=>(0.02,0.36),"c"=>(-75,-35),"d"=>(0.005,16))#,"I"=>[100,9000])
    lower,upper = get_ranges(ranges_izhi)
    return lower,upper
end

function initd()
    population = initf(50)
    garray = zeros((length(population)[1], length(population[1])))
    for (i,p) in enumerate(population)
        garray[i,:] = p
    end
    garray[1,:]
end
function get_adexp_ranges()
    ranges_adexp = DataStructures.OrderedDict{String,Tuple{Float32,Float32}}()
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
    lower,upper = get_ranges(ranges_adexp)
    return lower,upper
end

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
