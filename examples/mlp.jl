# https://github.com/joshday/OnlineStats.jl/blob/master/notebooks/animations.jl
#using UnicodePlots
using Flux
using Printf
using Plots
using Distributions: Uniform, Normal
using Statistics: mean
using Base.Iterators: partition
using Random: randperm, seed!
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
SNN.@load_units
#using MLJ
using Markdown
using InteractiveUtils
using JLD
using Pkg
using PyCall
using Flux
using PyPlot
using Zygote
##
# add zygote gradient tracking.
##
# https://discourse.julialang.org/t/multilayer-perceptron-with-multidimensional-output-array-using-flux/69747/2
dir(x) = fieldnames(typeof(x))
singlify(x::Float64) = Float32(x)
predict(x, m) = m.W*x .+ m.b

filename = string("../JLD/PopulationScatter_adexp.jld")

trace = load(filename , "trace")
evo_population = [t.metadata["pop"] for t in trace]
fitpop = [t.metadata["fitpop"] for t in trace]
singlify(x::Float64) = Float32(x)
x = []#evo_population
y = []#evo_loss
refitpop = []
rex = []
for (i,l) in enumerate(evo_population)
	if i>1
		l[1] = singlify.(l[1])
		append!(x,l[1])
		append!(refitpop,fitpop[i])

	end
end

y = refitpop
long_error=[]
for f in refitpop
	for i in f
		append!(long_error,i[1])
	end
end
y = singlify.(y)
data = Flux.Data.DataLoader((x,y), batchsize=4)#,shuffle=true);
#data = Flux.Data.DataLoader((x,y), batchsize=4,shuffle=true); # automatic batching convenience
X1, Y1 = data#first(data);
@show size(X1) size(Y1)
#X2, Y2 = data
#@show size(X1) size(Y1)

Nx = size(X1, 1)
Ny = size(Y1, 1)
model = Chain(Dense(Nx, Nx, sigmoid), Dense(Nx, Ny, sigmoid), Dense(Ny, Ny, identity));
#gs = gradient(() -> sum(model(X1)), params(model))
# Loss/cost function
loss(x, y) = Flux.mse(model(x), y)
loss(X1,Y1)
# Parameters of the model
para = Flux.params(model);
trnloss = [];
opt = ADAM()

for epoch = 1:500
	@show(size(data))
	@show(size(X1))

    Flux.train!(loss,para,data,opt)
    append!(trnloss, loss(X1, Y1))
	@show(trnloss)
end

amod = model(last(keys(opt.state)))
amod.W
predict(x, m) = m.W*x .+ m.b

gs[amod.W]

mach = machine(model, X1, Y1)
evaluate!(mach, resampling=cv, measure=l2, verbosity=0)

ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms

function nloss(E,ngt_spikes,ground_spikes)
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



function nloss(param)
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
