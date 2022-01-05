using SpikingNeuralNetworks
using Plots
SNN = SpikingNeuralNetworks
SNN.@load_units
using SpikeNetOpt
SNO = SpikeNetOpt
@testset "IZHI_spike_search" begin

    cell_type = "IZHI"
    ngt_spikes = rand(5:15)
    RS = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    current_ = SNO.current_search(cell_type, RS, ngt_spikes)
    RS.I = [current_ * nA]
    SNN.monitor(RS, [:v])
    SNN.monitor(RS, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms

    SNN.sim!(
        [RS];
        dt = 1 * ms,
        delay = ALLEN_DELAY,
        stimulus_duration = ALLEN_DURATION,
        simulation_duration = ALLEN_DURATION + ALLEN_DELAY + 443ms,
    )

    spikes = SNO.get_spikes(RS)
    spikes = [s * ms for s in spikes]
    nspk = size(spikes)[1]
    @test nspk == ngt_spikes

end
