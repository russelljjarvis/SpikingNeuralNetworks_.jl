using Distributed

if nprocs()==1
	addprocs(8)
end
@everywhere include("spike_distance_opt.jl")

#Random.seed!(0);
result = Evolutionary.optimize(
    loss,
    lower,
    upper,
    initd,
    options,
    Evolutionary.Options(
        iterations = 125,
        successive_f_tol = 75,
        show_trace = true,
        store_trace = true,
    ),
)
fitness = minimum(result)
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
evo_loss = [t.value for t in trace]
display(plot(evo_loss))
E1, spkd_found = eval_best(params)

#first_dim1 = [t.metadata["population"][1][1] for t in trace]
#first_dim2 = [t.metadata["population"][1][2] for t in trace]
#first_dim3 = [t.metadata["population"][1][3] for t in trace]
#first_dim4 = [t.metadata["population"][1][4] for t in trace]

#display(plot(first_dim1))
#display(plot(first_dim1,first_dim2,first_dim3))

run(`python-jl validate_candidate.py`)
