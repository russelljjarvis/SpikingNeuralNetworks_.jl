# https://github.com/joshday/OnlineStats.jl/blob/master/notebooks/animations.jl
#using UnicodePlots
using Markdown
using InteractiveUtils
using JLD
using Pkg
using PyCall
using PyPlot

trace = load("JLD/PopulationScatter.jld", "trace")
evo_population = [t.metadata["pop"] for t in trace]
evo_loss = [t.value for t in trace]
evo_iteration = [t.iteration for t in trace]
points = last(evo_population)

using PyCall
@pyimport matplotlib.animation as anim
using PyPlot

function showanim(filename)
    base64_video = base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end

fig = figure(figsize=(4,4))
ax = axes()

anim = @animate for (points, j, k) in zip(evo_population, evo_loss, evo_iteration)
    X = points[1,:]
    Y = points[2,:]
    Z = points[3,:]

    #scatter(X, Y, Z)
    show(PyPlot.scatter(X, Y, c=[j for i in 1:4]))#,markersize=j))#,markercolor=j))
    show(PyPlot.scatter(Z, Y, c=[j for i in 1:4]))#,markersize=j))#,markercolor=j))
    show(PyPlot.scatter(X, Z, c=[j for i in 1:4]))#,markersize=j))#,markercolor=j))

    #scatter(x,y, marker_z = z, markersize = 5*z,  color = :jet)
    #using VegaLite
    #display(@vlplot(:point, x=:X, y=:Y, color=:j))
    #cnt+=1
end
gif(anim, joinpath(directed, "evo.gif"), fps = 10)


