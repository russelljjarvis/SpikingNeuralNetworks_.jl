using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
SNN.@load_units
using SignalAnalysis

function get_spikes(p)
    fire = p.records[:fire]
    spikes = Float32[]
    for time = eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(spikes,time)
        end
    end
    spikes
end



function test_current(cell_type,param,current,ngt_spikes)
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

    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])

    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms

    E.I = [current*nA]
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+343ms)
    spikes = get_spikes(E)
    spikes = [s/1000 for s in spikes]
    nspk = size(spikes)[1]
    delta = abs(nspk - ngt_spikes)
    return nspk
end

#current_dict = test_c(check_values,param,ngt_spikes,current_dict,cell_type)

function test_c(check_values,param,ngt_spikes,current_dict,cell_type)
    nspk = -1.0
    for i_c in check_values
        nspk = test_current(cell_type,param,i_c,ngt_spikes)
        current_dict[nspk] = i_c
        if nspk == ngt_spikes
            if ngt_spikes in keys(current_dict)
                return current_dict
            end
        end
    end
    return current_dict
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


function current_search(cell_type,param,ngt_spikes)

    current_dict = Dict()
    new_bottom = minc =0
    new_top = maxc = 1000.0
    step_size = (maxc-minc)/10.0
    check_values = minc:step_size:maxc
    current_dict = Dict()
    cnt = 0.0
    while (ngt_spikes in keys(current_dict))==false
        current_dict = test_c(check_values,param,ngt_spikes,current_dict,cell_type)
        if ngt_spikes in keys(current_dict)
            return current_dict[ngt_spikes]
        end
        over_s = Dict([(k,v) for (k,v) in current_dict if k>ngt_spikes])
        under_s = Dict([(k,v) for (k,v) in current_dict if k<ngt_spikes])
        if length(over_s)>0
            # find the lowest current that caused more than one spike
            # throw away part of dictionary that has no spikes
            # find minimum of value in the remaining dictionary
            new_top = findmin(collect(values(over_s)))[1]

        else
            new_top = new_top*2.0
        end

        if length(under_s)>0
            new_bottom = findmax(collect(values(under_s)))[1]

        else
            new_bottom = (1.0/2.0)*abs(new_bottom)

        end
        step_size = abs(new_top-new_bottom)/15.0
        if step_size==0

            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            return closest

        end
        check_values = new_bottom:step_size:new_top

        cnt+=1
        if cnt >100
            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            return closest
        end
    end
    return current_dict[ngt_spikes]
end
