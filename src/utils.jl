using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
using Evolutionary, Test, Random

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



function get_ranges(ranges)
    ###
    # Code appropriated from:
    # https://github.com/JuliaML/MLPlots.jl/blob/master/src/optional/onlineai.jl
    ###

    lower = Float32[]
    upper = Float32[]
    for (k, v) in ranges
        append!(lower, v[1])
        append!(upper, v[2])
    end
    lower, upper
end

function init_b(lower, upper)
    gene = [] # TODO: should be a typed list one day
    for (i, (l, u)) in enumerate(zip(lower, upper))
        p1 = rand(l:u, 1)
        append!(gene, p1)
    end
    gene
end

function initf(n)
    genesb = []  # TODO: should be a typed list one day
    for i = 1:n
        genes = init_b(lower, upper)
        append!(genesb, [genes])
    end
    genesb
end

function custom_raster(P1::Array, P2::Array)
    y0 = Int32[0]
    X = Float32[]
    Y = Float32[]
    N = 1
    y = 0
    for p in [P1, P2]
        y += 1
        append!(X, P1)
        append!(Y, 1 .+ sum(y0))
        push!(y0, 1)
        append!(X, P2)
        append!(Y, 1 .+ sum(y0))
        push!(y0, 1)
        N += 1
    end
    plt = scatter(
        X,
        Y,
        w = 50,
        m = (1, :black),
        leg = :none,
        marker = :vline,
        xaxis = ("t", (0, Inf)),
        yaxis = ("neuron",),
    )
    y0 = y0[2:end-1]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
    return plt
end


function get_vm(p, simulation_duration)
    vm = p.records[:v]
    vm = [i[1] for i in vm]
    vm = signal(vm, length(vm) / simulation_duration)
    vm
end

function checkmodel(param, cell_type, ngt_spikes)
    if cell_type == "IZHI"
        pp = SNN.IZParameter(; a = param[1], b = param[2], c = param[3], d = param[4])
        E = SNN.IZ(; N = 1, param = pp)
    end
    if cell_type == "ADEXP"
        adparam = SNN.ADEXParameter(;
            a = param[1],
            b = param[2],
            cm = param[3],
            v0 = param[4],
            τ_m = param[5],
            τ_w = param[6],
            θ = param[7],
            delta_T = param[8],
            v_reset = param[9],
            spike_delta = param[10],
        )
        E = SNN.AD(; N = 1, param = adparam)
    end


    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    current = current_search(cell_type, param, ngt_spikes)
    E.I = [current * nA]

    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])

    ###
    # TODO buid t into neuron models
    # SNN.monitor(E, [:t])
    ###
    simulation_duration = ALLEN_DURATION + ALLEN_DELAY + 343ms
    SNN.sim!(
        [E];
        dt = 1 * ms,
        delay = ALLEN_DELAY,
        stimulus_duration = ALLEN_DURATION,
        simulation_duration = simulation_duration,
    )
    spikes = get_spikes(E)
    spikes = [s / 1000.0 for s in spikes]
    vm = get_vm(E, simulation_duration)
    (vm, spikes)
end

#=
function Evolutionary.value!(
    ::Val{:multiproc},
    fitness,
    objfun,
    population::AbstractVector{IT},
) where {IT}
    fitness = SharedArrays.SharedArray{Float32}(fitness)
    @time @sync @distributed for i = 1:length(population)
        fitness[i] = value(objfun, population[i])
    end
    fitness
end
=#
function Evolutionary.value!(
    ::Val{:serial},
    fitness,
    objfun,
    population::AbstractVector{IT},
) where {IT}
    Threads.@threads for i = 1:length(population)
        fitness[i] = value(objfun, population[i])
    end
end
function get_data()
    file = "../JLD/ground_truth.jld"
    if file
        vmgtv = load(file, "vmgtv")
        ngt_spikes = load(file, "ngt_spikes")
        gt_spikes = load(file, "gt_spikes")

        ground_spikes = gt_spikes
        ngt_spikes = size(gt_spikes)[1]
        vmgtt = load(file, "vmgtt")

        plot(plot(vmgtv, vmgtt, w = 1))

    else
        # if the file doesn't exist complete the expensive operation of making the file.
        # using Python
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
        julia_version_vm = signal(vmgtv, length(vmgtt) / last(vmgtt))
        plot(plot(vmgtv, vmgtt, w = 1))

        save(
            "ground_truth.jld",
            "vmgtv",
            vmgtv,
            "vmgtt",
            vmgtt,
            "ngt_spikes",
            ngt_spikes,
            "gt_spikes",
            gt_spikes,
            "julia_version_vm",
            julia_version_vm,
        )
    end

    vmgtv = load("ground_truth.jld", "vmgtv")
    ngt_spikes = load("ground_truth.jld", "ngt_spikes")
    ngt_spikes = size(gt_spikes)[1]
    ground_spikes = load("ground_truth.jld", "gt_spikes")

    vmgtt = load("ground_truth.jld", "vmgtt")
    julia_version_vm = load("ground_truth.jld", "julia_version_vm")
    return (vmgtv, vmgtt, ngt_spikes, ground_spikes, julia_version_vm)
end
function get_izhi_ranges()
    ranges_izhi = DataStructures.OrderedDict{Char,Float32}()
    ranges_izhi =
        ("a" => (0.002, 0.3), "b" => (0.02, 0.36), "c" => (-75, -35), "d" => (0.005, 16))#,"I"=>[100,9000])
    lower, upper = get_ranges(ranges_izhi)
    return lower, upper
end

function get_adexp_ranges()
    ranges_adexp = DataStructures.OrderedDict{String,Tuple{Float32,Float32}}()
    ranges_adexp[:"a"] = (2.0, 10)
    ranges_adexp[:"b"] = (5.0, 10)
    ranges_adexp[:"cm"] = (700.0, 983.5)
    ranges_adexp[:"v0"] = (-70, -55)
    ranges_adexp[:"τ_m"] = (10.0, 42.78345)
    ranges_adexp[:"τ_w"] = (300.0, 454.0)  # Tau_w 0, means very low adaption
    ranges_adexp[:"θ"] = (-45.0, -10)
    ranges_adexp[:"delta_T"] = (1.0, 5.0)
    ranges_adexp[:"v_reset"] = (-70.0, -15.0)
    ranges_adexp[:"spike_delta"] = (1.25, 20.0)
    lower, upper = get_ranges(ranges_adexp)
    return lower, upper
end

function vecplot(p, sym)
    v = SNN.getrecord(p, sym)
    y = hcat(v...)'
    x = 1:length(v)
    plot(x, y, leg = :none, xaxis = ("t", extrema(x)), yaxis = (string(sym), extrema(y)))
end

function vecplot(P::Array, sym)
    plts = [vecplot(p, sym) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

function spike_train_difference(spkd_ground, spkd_found)
    if length(spkd_found) == 0
        return sum(ones(Int(length(spkd_ground))))
    end
    if length(spkd_found) > 0
        maxi0 = findmax(spkd_ground[:, :])[1]
        maxi1 = findmax(spkd_found[:, :])[1]
        maxi1 = maxi1[1]
        maxi0 = maxi0[1]
        mini = findmin([maxi0, maxi1])[1]
        l0 = length(spkd_ground)
        l1 = length(spkd_found)
        maxi = findmin([l0, l1])[1]
        spkd = zeros(Int(maxi))
        @inbounds for (_, i) in zip(spkd, eachindex(spkd))
            if !isempty(spkd_ground[i]) && !isempty(spkd_found[i])

                maxt1 = findmax(spkd_ground[i])[1]
                maxt2 = findmax(spkd_found[i])[1]
                maxt = findmax([maxt1, maxt2])[1]

                if maxt1 > 0.0 && maxt2 > 0.0
                    t, S = SpikeSynchrony.SPIKE_distance_profile(
                        unique(sort(spkd_ground[i])),
                        unique(sort(spkd_found[i]));
                        t0 = 0.0,
                        tf = maxt,
                    )
                    spkd[i] = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1]) # == SPIKE_distance(y1, y2)

                end
            else
                spkd[i] = 0.0
            end
        end
        scatter([i for i = 1:mini], spkd) |> display
        error = sum(spkd)
    end
end


function initd()
    population = initd(10)
    garray = zeros((length(population)[1], length(population[1])))
    for (i, p) in enumerate(population)
        garray[i, :] = p
    end
    garray[1, :]
end

function initd(n)
    genesb = []
    lower = Float32[0.0 0.0 0.0 0.0]
    upper = Float32[1.0 1.0 1.0 1.0]
    lower = vec(lower)
    upper = vec(upper)

    for i = 1:n
        genes = initd(lower, upper)
        append!(genesb, [genes])
    end
    genesb
end


function initd(lower, upper)
    gene = []

    for (i, (l, u)) in enumerate(zip(lower, upper))
        p1 = rand(l:u, 1)
        append!(gene, p1)
    end
    gene
end
