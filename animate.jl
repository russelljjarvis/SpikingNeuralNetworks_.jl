# https://github.com/joshday/OnlineStats.jl/blob/master/notebooks/animations.jl

using Markdown
using InteractiveUtils
using JLD
using Plots, Random
theme(:lime)
directed = pwd()

trace = load("PopulationScatter.jld", "trace")
evo_population = [t.metadata["population"] for t in trace]
evo_loss = [t.value for t in trace]
evo_iteration = [t.iteration for t in trace]
points = last(evo_population)

for (points, j, k) in zip(evo_population, evo_loss, evo_iteration)
    X = points[2, :]
    Y = points[3, :]
    Z = points[4, :]
    #println(k)
    #println(j)
    scatter(X, Y, Z) |> display
end
anim = @animate for (points, j, k) in zip(evo_population, evo_loss, evo_iteration)
    X = points[1, :]
    Y = points[2, :]
    Z = points[3, :]
    scatter(X, Y, Z)
    #display(scatter(X, Y, Z))
end
gif(anim, joinpath(directed, "evo.gif"), fps = 1)


#md"![](firing_rates.png)"
