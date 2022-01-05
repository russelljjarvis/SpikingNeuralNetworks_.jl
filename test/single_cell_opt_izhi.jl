ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using SpikeNetOpt
import JLD

const ngt_spikes
const opt_vec
const extremum
const extremum_param
const ngt_spikes
const fitness

using Plots
###
const cell_type = "ADEXP"
global vecp = false
###
using SpikeNetOpt
SNO = SpikeNetOpt

(vmgtv, vmgtt, ngt_spikes, ground_spikes) = SNO.get_data()
println("Ground Truth")
plot(vmgtt[:], vmgtv[:]) |> display

lower, upper = get_adexp_ranges()
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms

function loss(E, ngt_spikes, ground_spikes)
    spikes = raster_synchp(E)
    spikes = [s * ms for s in spikes]
    maxt = findmax(sort!(unique(vcat(spikes, ground_spikes))))[1]
    if size(spikes)[1] > 1
        t, S = SPIKE_distance_profile(spikes, ground_spikes; t0 = 0, tf = maxt)
        spkd = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1]) # == SPIKE_distance(y1, y2)
    else
        spkd = 1.0
    end
    delta = abs(size(spikes)[1] - ngt_spikes)
    return spkd + delta

end



function zz_(param)
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

    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    current = current_search(cell_type, param, ngt_spikes)
    E.I = [current * nA]#[param[5]*nA]

    SNN.sim!(
        [E];
        dt = 1 * ms,
        delay = ALLEN_DELAY,
        stimulus_duration = ALLEN_DURATION,
        simulation_duration = ALLEN_DURATION + ALLEN_DELAY + 443ms,
    )
    vecplot(E, :v) |> display
    error = loss(E, ngt_spikes, ground_spikes)
    error
end


function z(x::AbstractVector)
    param = x
    params = SNN.IZParameter(; a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(; N = 1, param = params)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [param[5] * nA]
    SNN.sim!(
        [E];
        dt = 1 * ms,
        delay = ALLEN_DELAY,
        stimulus_duration = ALLEN_DURATION,
        simulation_duration = ALLEN_DURATION + ALLEN_DELAY + 500ms,
    )
    error = loss(E, ngt_spikes)
    @show(error)
    error

end


function checkmodel(param)
    if cell_type == "IZHI"
        pp = SNN.IZParameter(; a = param[1], b = param[2], c = param[3], d = param[4])
        E = SNN.IZ(; N = 1, param = pp)
    end
    if cell_type == "ADEXP"

        adparam = SNN.ADEXParameter(;
            a = param[1],
            b = param[2],
            cm = param[3],
            v_rest = param[4],
            tau_m = param[5],
            tau_w = param[6],
            v_thresh = param[7],
            delta_T = param[8],
            v_reset = param[9],
            spike_height = param[10],
        )
    end
    E = SNN.AD(; N = 1, param = adparam)


    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [current_search(param, ngt_spikes) * nA]

    SNN.monitor(E, [:v])
    SNN.sim!(
        [E];
        dt = 1 * ms,
        delay = ALLEN_DELAY,
        stimulus_duration = ALLEN_DURATION,
        simulation_duration = ALLEN_DURATION + ALLEN_DELAY + 443ms,
    )
    if vecp
        vec = SNN.vecplot(E, :v)
        vec |> display
        vec
    end

end

ɛ = 0.125
options = GA(
    populationSize = 20,
    ɛ = ɛ,
    selection = ranklinear(1.5),
    crossover = intermediate(1.0),
    mutation = uniform(1.0),
)
@time result = Evolutionary.optimize(
    zz_,
    lower,
    upper,
    initd,
    options,
    Evolutionary.Options(
        iterations = 20,
        successive_f_tol = 25,
        show_trace = true,
        store_trace = true,
    ),
)
fitness = minimum(result)
println("GA: ɛ=$ɛ) => F: $(minimum(result))")
extremum_param = Evolutionary.minimizer(result)
opt_vec = checkmodel(extremum_param)
println("probably jumbled extremum param")
println(extremum_param)
plot(opt_vec) |> display
plot(vmgtt[:], vmgtv[:]) |> display
