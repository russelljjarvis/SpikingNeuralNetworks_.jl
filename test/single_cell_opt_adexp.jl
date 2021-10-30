ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Pkg
using PyCall
using OrderedCollections
using LinearAlgebra
using SpikeSynchrony
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
SNN.@load_units
using Distributed
import DataStructures
using JLD

global ngt_spikes
global opt_vec
global extremum
global extremum_param
global ngt_spikes
global fitness

using Plots
unicodeplots()




PARALLEL=false
if PARALLEL
	if nprocs()==1
		addprocs(8)

	end
	@everywhere include("../opt_single_cell_utils.jl")

	function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
	    fitness = SharedArrays.SharedArray{Float32}(fitness)
	    @time @sync @distributed for i in 1:length(population)
	        fitness[i] = value(objfun, population[i])
	    end
	    @show(fitness)
	    fitness
	end
else
	include("../opt_single_cell_utils.jl")
end



###
global cell_type="ADEXP"
global vecp=false
###

(vmgtv,vmgtt,ngt_spikes,ground_spikes) = get_data()
println("Ground Truth")
plot(vmgtt[:],vmgtv[:]) |> display

#lower,upper=get_izhi_ranges()

lower,upper=get_adexp_ranges()
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms

function loss(E,ngt_spikes,ground_spikes)
    spikes = raster_synchp(E)
    spikes = [s*ms for s in spikes]
	@show(spikes)
	maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkdistance = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkdistance = 10.0
    end
	@show(spkdistance)
    delta = abs(size(spikes)[1] - ngt_spikes)
    return spkdistance+delta

end



function zz_(param)
    current = current_search(cell_type,param,ngt_spikes)
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
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    vecplot(E, :v) |> display
    error = loss(E,ngt_spikes,ground_spikes)
	@show(error)

    error
end

#=
function z(x::AbstractVector)
    param = x
    params = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = params)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [param[5]*nA]
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    error = loss(E,ngt_spikes)
    @show(error)
    error

end
=#
ɛ = 0.125
options = GA(
    populationSize = 25,
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
#plot(opt_vec)|> display
#plot(vmgtt[:],vmgtv[:])|> display
p1=plot(opt_vec)
plot!(p1,vmgtt[:],vmgtv[:])|> display
#, label = "randData", ylabel = "Y axis",color = :red, legend = :topleft, grid = :off, xlabel = "Numbers Rand")
#p = twiny()
#plot!(p,vec, label = "log(x)", legend = :topright, box = :on, grid = :off, xlabel = "Log values") |> display
#return fitness
#end
#end
#break
#end
