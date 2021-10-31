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
using Metaheuristics

using Plots
unicodeplots()




PARALLEL=false
if PARALLEL
	if nprocs()==1
		addprocs(8)

	end
	@everywhere include("../utils.jl")
	@everywhere include("../current_search.jl")

	eval(macroexpand(quote @everywhere using SpikingNeuralNetworks end))
	eval(macroexpand(quote @everywhere using PyCall end))

	function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
	    fitness = SharedArrays.SharedArray{Float32}(fitness)
	    @time @sync @distributed for i in 1:length(population)
	        fitness[i] = value(objfun, population[i])
	    end
	    @show(fitness)
	    fitness
	end
else
	include("../utils.jl")
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
    spikes = [s/1000.0 for s in spikes]
	#@show(spikes)
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



function loss(param)
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

ɛ = 0.125
options = GA(
    populationSize = 40,
    ɛ = ɛ,
    selection = ranklinear(1.5),#ss,
    crossover = intermediate(1.0),#line(1.0),#xovr,
    mutation = uniform(1.0),#domainrange(fill(1.0,ts)),#ms
)
@time result = Evolutionary.optimize(loss,lower,upper, initd, options,
    Evolutionary.Options(iterations=40, successive_f_tol=25, show_trace=true, store_trace=true)
)
fitness = minimum(result)
println("GA: ɛ=$ɛ) => F: $(minimum(result))")# C: $(Evolutionary.iterations(result))")
extremum_param = Evolutionary.minimizer(result)


opt_vec = checkmodel(extremum_param,cell_type,ngt_spikes)

#@show(result)
#@show(result.trace)
trace = result.trace
dir(x) = fieldnames(typeof(x))
dir(trace[1, 1, 1])
trace[1, 1, 1].metadata#["population"]
filename = string("../JLD/PopulationScatter_adexp.jld")#, py"target_num_spikes")#,py"specimen_id)
#save(filename, "trace", trace)
save(filename, "opt_vec", opt_vec,"extremum_param", extremum_param, "vmgtt", vmgtt, "vmgtv", vmgtv)

#evo_population = [t.metadata[""] for t in trace]
#E1, spkd_found = eval_best(params)

evo_loss = [t.value for t in trace]

display(plot(evo_loss))
println("probably jumbled extremum param")
println(extremum_param)

p1=plot(opt_vec) |> display
plot!(p1,vmgtt[:],vmgtv[:])|> display
plot(vmgtt[:],vmgtv[:]) |> display

@show(result.minimizer)

@show(fitness)

#plot(opt_vec)|> display
#plot(vmgtt[:],vmgtv[:])|> display

#=
D = 10
bounds = [3ones(D) 40ones(D)]'

bounds = [lower upper]'
a = view(bounds, 1, 1)
b = view(bounds, 1, 2)

information = Information(f_optimum = 0.0)
options = Options( seed = 1, iterations=10, f_calls_limit =10)

D = size(bounds, 2)
nobjectives=1
methods = [
        SMS_EMOA(N = 20, n_samples=20, options=options),
        NSGA2(options=options),
        #MOEAD_DE(gen_ref_dirs(1, 1), options=Options( seed = 1, iterations = 5)),
        #NSGA3(options=options),
      ]

#for method in [methods[1]]
    #f_calls = 0
result = ( optimize(loss, bounds, NSGA2(options=options)) )
show(IOBuffer(), "text/html", result)
show(IOBuffer(), "text/plain", result.population)
show(IOBuffer(), "text/html", result.population)
show(IOBuffer(), result.population[1])
#end
#, label = "randData", ylabel = "Y axis",color = :red, legend = :topleft, grid = :off, xlabel = "Numbers Rand")
#p = twiny()
#plot!(p,vec, label = "log(x)", legend = :topright, box = :on, grid = :off, xlabel = "Log values") |> display
#return fitness
#end
#end
#break
#end
=#
