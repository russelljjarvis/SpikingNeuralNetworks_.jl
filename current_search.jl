SNN = SpikingNeuralNetworks
SNN.@load_units
#nspk = test_current(cell_type,param,i_c,ngt_spikes)
#ERROR: LoadError: MethodError: no method matching
#test_current(::String, ::SpikingNeuralNetworks.IZ{Array{Float32,1},Array{Bool,1}}, ::Float64, ::Int64)
function raster_synchp(p)
    #show(p)
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
function test_current(cell_type,param,current,ngt_spikes)
    #param = param.param
    #println(param)
    #@show(param)
    if cell_type=="IZHI"
        #println(fields(typeof(param)))
        dir(x) = fieldnames(typeof(x))
        param = param.param
        #println(dir(param))
        #param[1]
        #a = param[:a]
        #pp = SNN.IZParameter(;a = param["a"], b = param["b"], c = param["c"], d = param["d"])

        #pp = SNN.IZParameter(;a = 0.1, b = 0.26, c = -65, d = 2)
        pp = SNN.IZParameter(;a = 0.02, b = 0.2, c = -65, d = 8)

        #@show(pp)
        #SNN.sim!(pp, [], 1ms)

        E = SNN.IZ(;N = 1, param = pp)
        #@show(E)
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

    E.I = [current*nA]#[param[5]*nA]

    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    #SNN.vecplot(E, :v) |> display
    spikes = raster_synchp(E)
    spikes = [s*ms for s in spikes]
    nspk = size(spikes)[1]
    delta = abs(nspk - ngt_spikes)
    @show(delta)
    @show(nspk)
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
        #@show(keys(current_dict))
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
            new_top = new_top*2.0
            #@show(new_top)
            #println("gets here")
        end

        if length(under_s)>0
            #store_min=collect(values(under_s))
            #@show(store_min)
            #@show(under_s)
            new_bottom = findmax(collect(values(under_s)))[1]
            #@show(new_bottom)

        else
            new_bottom = (1.0/2.0)*abs(new_bottom)

        end
        #@show(new_bottom)
        step_size = abs(new_top-new_bottom)/15.0
        #@show(step_size)
        if step_size==0

            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            return closest

        end
        check_values = new_bottom:step_size:new_top
        #println(new_bottom,new_top)
        @show(new_bottom)

        @show(new_top)
        @show(check_values)

        cnt+=1
        #check_values = 2.467264e6:
        #57.6:
        #2.46784e6

        if cnt >20
            println("intended number spikes $ngt_spikes")
            #@show(current_dict)
            #println(current_dict)
            flt = convert(Float32, ngt_spikes)
            tmp = collect(values(current_dict))
            closest = findnearest(tmp[:],ngt_spikes)#real(ngt_spikes))
            #println("spike disparity $closest")
            return closest
            #findmin(collect(values(current_dict)))[1]
        end
    end
    #@show(current_dict)
    return current_dict[ngt_spikes]
end
