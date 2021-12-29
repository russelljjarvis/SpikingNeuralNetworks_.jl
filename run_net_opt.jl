using Distributed
using ClearStacktrace

#using UnicodePlots
#using Plots
#unicodeplots()

#if nprocs()==1
#	addprocs(8)
#end
#@everywhere include("spike_distance_opt.jl")
include("spike_distance_opt.jl")

P, C = make_net_SNN(26)

E, I = P #, EEA]
EE, EI, IE, II = C
SNN.monitor([E, I], [:fire])
#global E_stim = []#Vector
sim_length = 500
@inbounds for t = 1:sim_length*ms
    E.I = vec([11.5 for i = 1:sim_length*ms])#vec(E_stim[t,:])#[i]#3randn(Ne)
    SNN.sim!(P, C, 1ms)

end
#_,_,_,spkd_ground = raster_synchp(P[1])
spkd_ground = get_trains(P[1])
sgg = [convert(Array{Float32,1}, sg) for sg in spkd_ground]

#MU = 10
#ɛ = MU / 2#0.125
#parallelization = :multi,
#f(x) = 10length(x) + sum( x.^2 - 10cos.(2π*x)  )

#Instantiate the bounds, note that bounds should be a $2\times 10$ Matrix where the first row corresponds to the lower bounds whilst the second row corresponds to the upper bounds.

#D = 10
#nobjectives = length(pf[1].f)
#npartitions = nobjectives == 2 ? 100 : 12

#methods = [
#        SMS_EMOA(N = 50, n_samples=500, options=options),
#        NSGA2(options=options),
#        MOEAD_DE(gen_ref_dirs(nobjectives, npartitions), options=Options( seed = 1, iterations = 500)),
#        NSGA3(options=options),
#      ]

#for method in methods

#f_calls = 0
f(x) = begin
    f_calls += 1
    loss(x)
end
f_calls = 0
#E1, spkd_found = eval_best(params)
#errors = loss(10)
#D = length(loss(10))
#@show(errors)
#bounds = Matrix([3.0; 40.0])
#bounds = [ 3.0; 40.0 ]'
D = 10
bounds = [3ones(D) 40ones(D)]'
a = view(bounds, 1, 1)
b = view(bounds, 1, 2)
#b = problem.bounds[2]
#@show(b)
#@show(a)
#@show(method)
#options = Options( seed = 1, iterations=10, f_calls_limit =10)
#N = 50, n_samples=500,
#method = NSGA2(options=options, f_calls_limit =100)#, information = information)
@show(bounds)
@show(loss)

information = Information(f_optimum = 0.0)
method = ECA(options=options, information = information)
result = optimize(loss, bounds, method)
@show(result)
#Approximate the optimum using the function optimize.

#result = optimize(loss, bounds)
#=
options = GA(
    populationSize = MU,
    ɛ = 4,
    mutationRate = 0.5,
    selection = ranklinear(1.5),#ranklinear(1.5),#ss,
    crossover = intermediate(0.5),#xovr,
    mutation = uniform(0.5),#(.015),#domainrange(fill(1.0,ts)),#ms
)

#Random.seed!(0);
result = Evolutionary.optimize(
    loss,
    lower,
    upper,
    initd,
    options,
    Evolutionary.Options(
        iterations = 50,
        successive_f_tol = 1,
        show_trace = true,
        store_trace = true,
    ),
)
fitness = minimum(result)
=#
#parallelization = :thread,

filename = string("GAsolution.jld")#, py"target_num_spikes")#,py"specimen_id)
params = result.minimizer
E1, spkd_found = eval_best(params)
save(
    filename,
    "spkd_ground",
    spkd_ground,
    "spkd_found",
    spkd_found,
    "Ne",
    Ne,
    "Ni",
    Ni,
    "sim_length",
    sim_length,
)
println("best result")
loss(result.minimizer)
println("σee = 0.5,  pee= 0.8,σei = 0.5,  pei= 0.8")

#println("σee = 0.45,  pee= 0.8,σei = 0.4,  pei= 0.9)")
@show(result.minimizer)

@show(fitness)

@show(result)
@show(result.trace)
trace = result.trace
dir(x) = fieldnames(typeof(x))
dir(trace[1, 1, 1])
trace[1, 1, 1].metadata#["population"]
filename = string("PopulationScatter.jld")#, py"target_num_spikes")#,py"specimen_id)
save(filename, "trace", trace)
#evo_population = [t.metadata[""] for t in trace]
E1, spkd_found = eval_best(params)

evo_loss = [t.value for t in trace]

display(plot(evo_loss))

#first_dim1 = [t.metadata["population"][1][1] for t in trace]
#first_dim2 = [t.metadata["population"][1][2] for t in trace]
#first_dim3 = [t.metadata["population"][1][3] for t in trace]
#first_dim4 = [t.metadata["population"][1][4] for t in trace]

#display(plot(first_dim1))
#display(plot(first_dim1,first_dim2,first_dim3))

#run(`python-jl validate_candidate.py`)
using PyCall
py"""import validate_candidate"""
iter = [t.iteration for t in trace]
data = [ trace[i+1,1,1].metadata["pop"] for i in iter ]

evo_loss


model = Chain(Dense(d, 15, relu), Dense(15, nclasses))

@info "MLP" loss=loss(data, evomodel) accuracy = accuracy(data, evomodel)
#model = Dense(2, 1, σ)
#L(x,y) = Flux.mse(model(x), y)
#opt = SGD(params(model))
#Flux.train!(L, zip(xs, ys), opt)
function loss(model)
    σee = model[1]
    pee = model[2]
    σei = model[3]
    pei = model[4]
    P1, C1 = make_net(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")

    SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground, spkd_found)
    error
end
#loss(model) = (x,y)->logitcrossentropy(model(x), y)
#loss(model,x,y) = loss(model)(x, y)
#loss(xy, model) = loss(model)(hcat(map(first,xy)...), hcat(map(last,xy)...))


opt = ADAM(1e-4)
evalcb = Flux.throttle(() -> @show(loss(data, model), accuracy(data, model)), 5)
for i in 1:500
    Flux.train!(loss(model), params(model), data, opt, cb = evalcb)
end
# ## Visualize the result

#contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[].data, fill=true)
