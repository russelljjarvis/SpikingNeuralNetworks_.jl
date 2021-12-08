#using Distributed
using ClearStacktrace

using Plots

using SpikeNetOpt
#@show(varinfo(SNO))
using JLD
using SpikingNeuralNetworks
using Evolutionary
using SpikingNN
using Metaheuristics
using Test


SNN = SpikingNeuralNetworks
SNN.@load_units
SNO = SpikeNetOpt

unicodeplots()
##
##


const MU = 10

##
# Ground truths
##
const weight_gain_factor = 0.77
#const GT = 26

#g, Cg = SpikeNetOpt.make_net_from_graph_structure(GT)

spkd_ground_dic = SNO.sim_net_darsnack(weight_gain_factor)#(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8)
spkd_ground = SNO.get_trains_dars(spkd_ground_dic)

error = SNO.spike_train_difference(spkd_ground, spkd_ground)
@test error==0
@show(error)
println(error==0)
weight_gain_factor1 = 0.77
nspkd_ground_dic = SNO.sim_net_darsnack(weight_gain_factor1)#(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8)
nspkd_ground = SNO.get_trains_dars(nspkd_ground_dic)
nerror = SNO.spike_train_difference(spkd_ground, nspkd_ground)
@test nerror==0

println(nerror)
@show(nerror)


META_HEUR_OPT = true
EVOLUTIONARY_OPT = false
function loss(model)
    @show(model)
    #println(model[1])
    #println(length(model))
    weight_gain_factor = model[1]
    train_dic = SNO.sim_net_darsnack(weight_gain_factor)
    spikes = SNO.get_trains_dars(train_dic)
    #display(Plots.historam(spikes))

    error = SNO.spike_train_difference(spkd_ground, spikes)
    error = sum(error)
    @show(error)
    error
end

if EVOLUTIONARY_OPT


    lower = Float32[0.0]#0.0 0.0 0.0]# 0.03 4.0]
    upper = Float32[3.0]# 1.0 1.0 1.0]# 0.2 20.0]
    lower = vec(lower)
    upper = vec(upper)

    options = GA(
        populationSize = MU,
        ɛ = 4,
        mutationRate = 0.5,
        selection = ranklinear(1.5),#ranklinear(1.5),#ss,
        crossover = intermediate(0.5),#xovr,
        mutation = uniform(0.5),#(.015),#domainrange(fill(1.0,ts)),#ms
    )
    #Random.seed!(0);
    D = 1
    bounds = [0.0ones(D) 3.0ones(D)]'
    result = Evolutionary.optimize(
        loss,
        lower,
        upper,
        vec([1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0]),
        options,
        Evolutionary.Options(
            iterations = 50,
            successive_f_tol = 1,
            show_trace = true,
            store_trace = true,
        ),
    )
    fitness = minimum(result)

    filename = string("GAsolution.jld")#, py"target_num_spikes")#,py"specimen_id)
    params = result.minimizer


    function eval_best(params)
        spkd_found = SNO.sim_net_darsnack(params[1])#Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)


        title!("Raster Plot")
        xlabel!("Time (sec)")
        spkd_found2 = SNO.get_trains_dars(spkd_found)
        display(Plots.historam(spkd_found2))
        spkd_found
        SpikingNN.rasterplot(spkd_found, label = ["Input 1"])#, "Input 2"])
    end

    spkd_found = eval_best(params)
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
    #println("σee = 0.5,  pee= 0.8,σei = 0.5,  pei= 0.8")

    #println("σee = 0.45,  pee= 0.8,σei = 0.4,  pei= 0.9)")
    @show(result.minimizer)

    @show(fitness)

    @show(result)
    @show(result.trace)
    trace = result.trace
    #SNO.dir(trace[1, 1, 1])
    trace[1, 1, 1].metadata#["population"]
    spkd_found = eval_best(params)
    evo_loss = [t.value for t in trace[2:length(trace)]]
    display(plot(evo_loss))
    filename = string("PopulationScatter.jld")#, py"target_num_spikes")#,py"specimen_id)
    save(filename, "trace", trace)

end
dir(x) = fieldnames(typeof(x))

#if META_HEUR_OPT
logger(st) = begin
    A = st.population
    temp = [a.x for a in A]
    bs = st.best_sol
    scatter(temp, label="opt", title="Gen: $(st.iteration)") |> display
    plot!([bs.x], label="Parento Front", lw=2) |> display
end
#function main()
       # Transliterated fortran main subroutine


D = 1

#bounds = [0.0ones(D);# lower bounds
#        1.0ones(D)]# upper bounds
bounds = Array([0.0ones(D) 1.1ones(D)]')
#pareto_set = [ generateChild(loss(bounds[i])) for i in 1:length(bounds) ]

#information = Information(f_optimum = 0.0)
options = Options( seed = 1, iterations=350, f_calls_limit =350)
#optimize(f, bounds, NSGA2())
information = Information(f_optimum =0.0)

algorithm = ECA(information = information, options = options)

result = optimize(loss, bounds, algorithm, logger=logger)#, algorithm)#, logger=logger)#, SMS_EMOA(N = 15, n_samples=1, options=options))
@show(result)
#end

#end

#main()

@show minimum(result)
@show minimizer(result)

f_calls, best_f_value = convergence(result)

animation = @animate for i in 1:length(result.convergence)
    l = @layout [a b]
    p = plot( layout=l)

    X = positions(result.convergence[i])
    scatter!(p[1], X[:,1], X[:,2], label="", xlim=(-5, 5), ylim=(-5,5), title="Population")
    x = minimizer(result.convergence[i])
    scatter!(p[1], x[1:1], x[2:2], label="")

    # convergence
    plot!(p[2], xlabel="Generation", ylabel="fitness", title="Gen: $i")
    plot!(p[2], 1:length(best_f_value), best_f_value, label=false)
    plot!(p[2], 1:i, best_f_value[1:i], lw=3, label=false)
    x = minimizer(result.convergence[i])
    scatter!(p[2], [i], [minimum(result.convergence[i])], label=false)
end

# save in different formats
# gif(animation, "anim-convergence.gif", fps=30)
mp4(animation, "anim-convergence.mp4", fps=30)
    #=
    result = optimize(loss, bounds, NSGA2(options=options))
    #result = optimize(SpikeNetOpt.loss, bounds)#, method)
    @show(result)

    methods = [
            SMS_EMOA(N = 5, n_samples=5, options=options),
            NSGA2(options=options)
            ]
            #MOEAD_DE(gen_ref_dirs(nobjectives, npartitions), options=Options( seed = 1, iterations = 500)),
            #NSGA3(options=options),


    for method in methods
        f_calls = 0
        result = optimize(loss, bounds, method)
        #result = optimize(SpikeNetOpt.loss, bounds)#, method)
        @show(result)

    end
    =#
#end


# ## Visualize the result

#contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[].data, fill=true)
