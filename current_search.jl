SNN = SpikingNeuralNetworks
SNN.@load_units

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

    E.I = [current*nA]#[param[5]*nA]
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)

    spikes = raster_synchp(E)
    spikes = [s*ms for s in spikes]

    #spikes = [s*ms for s in spikes]
    nspk = size(spikes)[1]
    delta = abs(nspk - ngs)
    return nspk
end
function test_c(check_values,param,ngt_spikes,current_dict)
    nspk = -1.0
    for i_c in check_values
        #@show(i_c)
        nspk = test_current(param,i_c,ngt_spikes)
        #@show(nspk)
        current_dict[nspk] = i_c
        if nspk == ngt_spikes
            if ngt_spikes in keys(current_dict)
                return current_dict,nspk
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


function current_search(param,ngt_spikes)

    current_dict = Dict()
    new_bottom = minc =100
    new_top = maxc = 200000.0
    step_size = (maxc-minc)/10.0
    check_values = minc:step_size:maxc
    current_dict = Dict()
    cnt = 0.0
    while (ngt_spikes in keys(current_dict))==false
        current_dict = test_c(check_values,param,ngt_spikes,current_dict)

        if ngt_spikes in keys(current_dict)
            return current_dict[ngt_spikes]
        end

        #@show(nspk)
        over_s = Dict([(k,v) for (k,v) in current_dict if k>ngt_spikes])
        under_s = Dict([(k,v) for (k,v) in current_dict if k<ngt_spikes])
        #@show(under_s)
        #if nspk>0
        if length(over_s)>0
            # find the lowest current that caused more than one spike
            # throw away part of dictionary that has no spikes
            # find minimum of value in the remaining dictionary
            new_top = findmin(collect(values(over_s)))[1]

        else
            new_top = new_top*10
            #@show(new_top)
            #println("gets here")
        end

        if length(under_s)>0
            #store_min=collect(values(under_s))
            #@show(store_min)
            new_bottom = findmax(collect(values(under_s)))[1]

        else
            new_bottom = (1.0/10.0)*abs(new_bottom)
        end
        #@show(new_bottom)
        #@show(new_top)

        #println(new_top)
        #println(new_bottom)
        #@show(new_top)
        #@show(new_bottom)

        step_size = abs(new_top-new_bottom)/10.0
        #@show(step_size)
        if step_size==0

            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            println("fails here")
            println("spike disparity $closest")
            return closest

        end
        check_values = new_bottom:step_size:new_top
        #println(new_bottom,new_top)
        @show(check_values)

        cnt+=1
        #check_values = 2.467264e6:
        #57.6:
        #2.46784e6

        if cnt >1000
            println("intended number spikes $ngt_spikes")
            @show(current_dict)
            #println(current_dict)
            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            println("spike disparity $closest")
            return closest
            #findmin(collect(values(current_dict)))[1]
        end
    end
    @show(current_dict)
    return current_dict[ngt_spikes]
end
