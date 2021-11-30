#using Distributed
using ClearStacktrace

using Plots
unicodeplots()

using SpikeNetOpt
SNO = SpikeNetOpt
@show(varinfo(SNO))

using SpikingNeuralNetworks
using Evolutionary
SNN = SpikingNeuralNetworks
SNN.@load_units

##
##


##
# Ground truths
##
global Ne = 200;
global Ni = 50
global σee = 1.0
global pee = 0.5
global σei = 1.0
global pei = 0.5
MU = 10


global E
global spkd_ground
global GT = 26


P, C = SpikeNetOpt.make_net(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8)
E, I = P
EE, EI, IE, II = C
SNN.monitor([E, I], [:fire])
sim_length = 1000
@inbounds for t = 1:sim_length
    E.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
    SNN.sim!(P, C, 1ms)

end
#spkd_ground = SpikeNetOpt.get(P[1])
spkd_ground = SpikeNetOpt.get_trains(P[1])
#@show(spkd_ground)
SNN.raster(P[1]) |> display
#sgg = [convert(Array{Float32,1}, sg) for sg in spkd_ground]
#Pg, Cg = SpikeNetOpt.make_net_from_graph_structure(GT)

#P, C = SpikeNetOpt.make_net(GT)
E, I = P
EE, EI, IE, II = C
SNN.monitor([E, I], [:fire])
sim_length = 500
@inbounds for t = 1:sim_length*ms
    E.I = vec([11.5 for i = 1:sim_length*ms])#vec(E_stim[t,:])#[i]#3randn(Ne)
    SNN.sim!(P, C, 1ms)

end
##
# Ground truth for optimization
##
spkd_ground = get_trains(P[1])
##
# Not really necessary
##
sgg = [convert(Array{Float32,1}, sg) for sg in spkd_ground]

META_HEUR_OPT = false
EVOLUTIONARY_OPT = true

if EVOLUTIONARY_OPT


    lower = Float32[0.0 0.0 0.0 0.0]# 0.03 4.0]
    upper = Float32[1.0 1.0 1.0 1.0]# 0.2 20.0]
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
        SpikeNetOpt.loss,
        lower,
        upper,
        SpikeNetOpt.initd,
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
