using SpikingNeuralNetworks
using ClearStacktrace
using Plots
unicodeplots()
SNN = SpikingNeuralNetworks
SNN.@load_units
#include("../current_search.jl")
using SpikeNetOpt
SNO = SpikeNetOpt
@testset "IZHI_spike_search" begin

    cell_type = "IZHI"
    ngt_spikes=rand(5:15)
    RS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -65, d = 8))

    current_ = SNO.current_search(cell_type,RS,ngt_spikes)

    RS.I = [current_*nA]
    SNN.monitor(RS, [:v])
    SNN.monitor(RS, [:fire])
    #test_result(nspk, ngt_spikes, 1e-1)

    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms

    SNN.sim!([RS]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)

    spikes = SNO.get_spikes(RS)
    spikes = [s*ms for s in spikes]
    nspk = size(spikes)[1]
    @test nspk==ngt_spikes


    #v = SNN.vecplot(RS, :v)
    #v |> display
end

#=
@testset "ADEXP_spike_search" begin

    cell_type = "ADEXP"
    ngt_spikes=rand(4:20)

    adparam = SNN.ADEXParameter(;a = 6.050246708405076, b = 7.308480222357973,
        cm = 803.1019662706587,
        v0= -63.22881649139353,
        τ_m=19.73777028610565,
        τ_w=351.0551915202058,
        θ=-39.232165554444265,
        delta_T=6.37124632135508,
        v_reset = -59.18792270568965,
        spike_delta = 16.33506432689027)



    E = SNN.AD(;N = 1, param=adparam)
    current_ = SNO.current_search(cell_type,E,ngt_spikes)
    E.I = [current_*nA]
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])

    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    spikes = SNO.raster_synchp(E)
    spikes = [s*ms for s in spikes]
    nspk = size(spikes)[1]
    #test_result(nspk, ngt_spikes, 1e-1)
    @test nspk==ngt_spikes

    #v = SNN.vecplot(E, :v)
    #v |> display
end
=#
#=
@testset "IZHI" begin

    RS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -65, d = 8))
    IB = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -55, d = 4))
    CH = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -50, d = 2))
    FS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.2, c = -65, d = 2))
    TC1 = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.25, c = -65, d = 0.05))
    TC2 = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.25, c = -65, d = 0.05))
    RZ = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.26, c = -65, d = 2))
    LTS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.25, c = -65, d = 2))
    P = [RS, IB, CH, FS, TC1, TC2, RZ, LTS]

    SNN.monitor(P, [:v])
    SNN.monitor(P, [:fire])

    T = 2second
    for t = 0:T
        for p in [RS, IB, CH, FS, LTS]
            p.I = [10]
        end
        TC1.I = [(t < 0.2T) ? 0mV : 2mV]
        TC2.I = [(t < 0.2T) ? -30mV : 0mV]
        RZ.I =  [(0.5T < t < 0.6T) ? 10mV : 0mV]
        SNN.sim!(P, [], 0.1ms)

        #test_result(nspk, ngt_spikes, 1e-1)

        #v = vecplot(E, :v)
        #@show(v)
    end
    for p in P
        spikes = raster_synchp(p)
        spikes = [s*ms for s in spikes]
        nspk = size(spikes)[1]
        @test nspk>=1

        v = SNN.vecplot(p, :v)
        v |> display
    end
end
=#
