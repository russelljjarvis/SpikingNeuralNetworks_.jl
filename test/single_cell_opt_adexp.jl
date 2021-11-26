ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
dir(x) = fieldnames(typeof(x))
singlify(x::Float64) = Float32(x)

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
using Metaheuristics
using Plots
unicodeplots()

global ngt_spikes
global opt_vec
global extremum
global extremum_param
global ngt_spikes
global fitness




PARALLEL=false
THREADS=false
if PARALLEL
	if nprocs()==1
		addprocs(8)
	else
		@everywhere include("../utils.jl")
		@everywhere include("../current_search.jl")
		eval(macroexpand(quote @everywhere using SpikingNeuralNetworks end))
		eval(macroexpand(quote @everywhere using PyCall end))

	end
else
	include("../utils.jl")
end



###
global cell_type="ADEXP"
global vecp=false
###

(vmgtv,vmgtt,ngt_spikes,ground_spikes,julia_version_vm) = get_data()
global simdur = last(vmgtt)*1000

println("Ground Truth")
plot(vmgtt[:],vmgtv[:]) |> display

#lower,upper=get_izhi_ranges()

lower,upper=get_adexp_ranges()
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms

function loss(E,ngt_spikes,ground_spikes)
    spikes = get_spikes(E)
    spikes = [s/1000.0 for s in spikes]

	#opt_vec = [i[1] for i in opt_vec]
	#s_a = signal(opt_vec, length(opt_vec)/last(vmgtt))
	#s_b = signal(vmgtv, length(vmgtt)/last(vmgtt))

	maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkdistance = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkdistance = 10.0
    end
	if length(spikes)>1

		custom_raster2(spikes,ground_spikes)
		custom_raster(spikes,ground_spikes)

		#savefig(crp, "aligned_VMs_adexp.png")
		#display(crp)
	end
	spkdistance*=spkdistance

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
	#@show(simdur)
    SNN.sim!([E]; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = simdur)
    vecplot(E, :v) |> display
    error = loss(E,ngt_spikes,ground_spikes)
	@show(error)

    error
end
#=
function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
	Threads.@threads for i in 1:length(population)
		fitness[i] = value(objfun, population[i])
	end
end
=#
ɛ = 0.125
options = GA(
    populationSize = 10,
    ɛ = ɛ,
    selection = ranklinear(1.5),#ss,
    crossover = intermediate(1.0),#line(1.0),#xovr,
    mutation = uniform(1.0),#domainrange(fill(1.0,ts)),#ms
)
@time result = Evolutionary.optimize(loss,lower,upper, initd, options,
    Evolutionary.Options(iterations=140, successive_f_tol=25, show_trace=true, store_trace=true)#,parallelisation=:thread)
)
import Plots
gr()

fitness = minimum(result)
println("GA: ɛ=$ɛ) => F: $(minimum(result))")# C: $(Evolutionary.iterations(result))")
extremum_param = Evolutionary.minimizer(result)

using SignalAnalysis
opt_vec,opt_spikes = checkmodel(extremum_param,cell_type,ngt_spikes)
trace = result.trace
filename = string("../JLD/PopulationScatter_adexp.jld")
save(filename,"trace",trace,"opt_vec", opt_vec,"extremum_param", extremum_param, "vmgtt", vmgtt, "vmgtv", vmgtv, "ground_spikes", ground_spikes,"opt_spikes",opt_spikes)
evo_loss = [t.value for t in trace]
loss_evolution = []
for (i,l) in enumerate(evo_loss)
	if i>1
		append!(loss_evolution,l)
	end
end
display(plot(loss_evolution))
println("Fittest model parameters")
println(extremum_param)

opt_vec = [i[1] for i in opt_vec]

s_a = signal(opt_vec, length(opt_vec)/last(vmgtt))
s_b = signal(vmgtv, length(vmgtt)/last(vmgtt))
p = plot(s_a)
p2 = plot!(p, s_b)
display(p2)
savefig(p2, "../imgs/aligned_VMs_adexp.png")


crp=custom_raster(opt_spikes,ground_spikes)
savefig(crp, "../imgs/aligned_VMs_adexp.png")
display(crp)

#display(plot(s_a))
#display(plot(s_b))

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
