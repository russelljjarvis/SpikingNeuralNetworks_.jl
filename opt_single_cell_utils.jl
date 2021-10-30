include("current_search.jl")

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

function get_data()
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
        plot(plot(vmgtv,vmgtt,w=1))

        save("ground_truth.jld", "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)
        filename = string("ground_truth: ", py"target_num_spikes")#,py"specimen_id)
        filename = string(filename,py"specimen_id")
        filename = string(filename,".jld")
        save(filename, "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)

    end

        vmgtv = load("ground_truth.jld","vmgtv")
        ngt_spikes = load("ground_truth.jld","ngt_spikes")
        ngt_spikes = size(gt_spikes)[1]

        ground_spikes = load("ground_truth.jld","gt_spikes")

        vmgtt = load("ground_truth.jld","vmgtt")
    return (vmgtv,vmgtt,ngt_spikes,ground_spikes)
end
function get_izhi_ranges()
    ranges_izhi = DataStructures.OrderedDict{Char,Float32}()
    ranges_izhi = ("a"=>(0.002,0.3),"b"=>(0.02,0.36),"c"=>(-75,-35),"d"=>(0.005,16))#,"I"=>[100,9000])
    lower,upper = get_ranges(ranges_izhi)
    return lower,upper
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
