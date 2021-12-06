#using Distributed
using ClearStacktrace

using Plots

using SpikeNetOpt
SNO = SpikeNetOpt
#@show(varinfo(SNO))

using SpikingNeuralNetworks
using Evolutionary
SNN = SpikingNeuralNetworks
SNN.@load_units
unicodeplots()

##
##


##
# Ground truths
##
MU = 10
const GT = 26

#g, Cg = SpikeNetOpt.make_net_from_graph_structure(GT)
weight_gain_factor = 0.77
spkd_ground = SNO.sim_net_darsnack(weight_gain_factor)#(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8)

META_HEUR_OPT = false
EVOLUTIONARY_OPT = true

function loss(model)
    @show(model)
    weight_gain_factor = model[1]
    spkd_found = SNO.sim_net_darsnack(weight_gain_factor)#,a=a)
    error = SNO.spike_train_difference(spkd_ground, spkd_found)
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
    result = Evolutionary.optimize(
        loss,
        lower,
        upper,
        SNO.initd,
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
        #σee = params[1]
        #pee = params[2]
        #σei = params[3]
        #pei = params[4]
        spkd_found = SNO.sim_net_darsnack(params[1])#Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
        #E1, I1 = P1
        #SNN.monitor([E1, I1], [:fire])
        #sim_length = 1000
        #@inbounds for t = 1:sim_length*ms
        #    E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        #    SNN.sim!(P1, C1, 1ms)
        #end

        #spkd_found = get_trains(P1[1])
        println("Ground Truth: \n")
        SNN.raster([E]) |> display
        println("candidate: \n")
        SNN.raster([E1]) |> display
        E1, spkd_found

    end

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
    SNO.dir(trace[1, 1, 1])
    trace[1, 1, 1].metadata#["population"]
    E1, spkd_found = eval_best(params)
    evo_loss = [t.value for t in trace[2:length(trace)]]
    #evo_loss = [t.value for t in trace[1:last(trace)]]
    display(plot(evo_loss))
    filename = string("PopulationScatter.jld")#, py"target_num_spikes")#,py"specimen_id)
    save(filename, "trace", trace)

end

if META_HEUR_OPT

    #lower = Float32[0.0 0.0 0.0 0.0]# 0.03 4.0]
    #upper = Float32[1.0 1.0 1.0 1.0]# 0.2 20.0]

    #lower = Int32[3]# 0.0 0.0 0.0]# 0.03 4.0]
    #upper = Int32[40]# 1.0 1.0 1.0]# 0.2 20.0]

    D = 10
    bounds = [3ones(D) 40ones(D)]'
    a = view(bounds, 1, 1)
    b = view(bounds, 1, 2)
    information = Information(f_optimum = 0.0)
    options = Options( seed = 1, iterations=10, f_calls_limit =10)

    D = size(bounds, 2)
    nobjectives=1
    #=
    methods = [
            SMS_EMOA(N = 5, n_samples=5, options=options),
            NSGA2(options=options),
            MOEAD_DE(gen_ref_dirs(1, 1), options=Options( seed = 1, iterations = 5)),
            NSGA3(options=options),
          ]

    for method in methods
        result = ( optimize(f, bounds, method) )
        show(IOBuffer(), "text/html", result)
        show(IOBuffer(), "text/plain", result.population)
        show(IOBuffer(), "text/html", result.population)
        show(IOBuffer(), result.population[1])
    end
    =#
    options = Options( seed = 1, iterations=10000, f_calls_limit = 25000)
    #nobjectives = length(pf[1].f)
    npartitions = nobjectives == 2 ? 100 : 12

    methods = [
            SMS_EMOA(N = 5, n_samples=5, options=options),
            NSGA2(options=options)
            ]
            #MOEAD_DE(gen_ref_dirs(nobjectives, npartitions), options=Options( seed = 1, iterations = 500)),
            #NSGA3(options=options),


    for method in methods
        f_calls = 0
        result = optimize(SpikeNetOpt.loss, bounds, method)
        #result = optimize(SpikeNetOpt.loss, bounds)#, method)
        @show(result)

    end

end


# ## Visualize the result

#contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[].data, fill=true)
