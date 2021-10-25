using SpikingNN
using Plots
using Distributions
T = 1000
# neuron parameters
vᵣ = 0
τᵣ = 1.0
vth = 1.0
weights = rand(Uniform(-2,1),25,25)
pop = gpu(Population(weights; cell = () -> LIF(τᵣ, vᵣ),
                          synapse = Synapse.Alpha,
                          threshold = () -> Threshold.Ideal(vth)))
# create input currents
low = ConstantRate(0.14)
high = ConstantRate(0.1499)
switch(t; dt = 1) = (t < Int(T/2)) ? low(t; dt = dt) : high(t; dt = dt)
n1synapse = QueuedSynapse(Synapse.Alpha())
n2synapse = QueuedSynapse(Synapse.Alpha())
excite!(n1synapse, filter(x -> x != 0, [low(t) for t = 1:T]))
excite!(n2synapse, filter(x -> x != 0, [switch(t) for t = 1:T]))
voltage_array_size = size(weights)[1]
voltages = Dict([(i, Float64[]) for i in 1:voltage_array_size])
cb = () -> begin
    for id in 1:size(pop)
        push!(voltages[id], getvoltage(pop[id]))
    end
end
input1 = [ (t; dt) -> 0 for i in 1:voltage_array_size/3]
input2 = [ n2synapse for i in voltage_array_size/3+1:2*voltage_array_size/3]
input3 = [ (t; dt) -> 0 for i in 2*voltage_array_size/3:voltage_array_size]
input = vcat(input2, input1, input3)
outputs = simulate!(pop, T; cb = cb, inputs=input)
labels = [ "input $i" for i in 1:voltage_array_size]
rasterplot(outputs, label = labels)#["Input 1", "Input 2","Input 3", "Inhibitor"])
title!("Raster Plot")
xlabel!("Time (sec)")
