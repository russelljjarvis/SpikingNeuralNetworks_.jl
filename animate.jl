# https://github.com/joshday/OnlineStats.jl/blob/master/notebooks/animations.jl

using Markdown
using InteractiveUtils

# ╔═╡ 2d679c48-8af5-11eb-32fc-c9a76a667e61
begin
	using Plots, Random
	theme(:lime)
end

md"![](firing_rates.png)"
n = 100
nframes = 100

begin
	#o = HeatMap(-5:.2:5, 0:.2:10)

	anim = @animate for i in 1:nframes
		x = randn(5i)
		y = randexp(5i)
		fit!(o, zip(x,y))
		plot(o)
	end
	gif(anim, "temp.gif", fps=10)
end


using Distributed
using ProgressMeter

@showprogress 1 "Computing..." for i in 1:50
    sleep(0.1)
end

@showprogress pmap(1:10) do x
    sleep(0.1)
    x^2
end

@showprogress reduce(1:10) do x, y
    sleep(0.1)
    x + y
end

function animatepopopt(dir::String)
  Plots.scalefontsizes()
  sctrpoplim = (args...;kwargs...) -> scatter(args...;kwargs...,xlims=(1, 100), ylims=(0,1), clims=(4, 9))
  sctroptlim = (args...;kwargs...) -> scatter(args...;kwargs...,xlims=(-6, 0), ylims=(0,1))

  scp = ScatterPop(sctrpoplim, dir);
  sco = ScatterOpt(sctroptlim, dir);
  l = @layout [a b]
  mpl = MultiPlot((ps...) -> plot(ps...; layout=(2,1), size=(600,800)), scp, sco; init=false)

  anim = @animate for i in eachindex(scp.data)
    pl = NaiveGAflux.plotgen(mpl, i)
    pl.subplots[1].attr[:title] = "Generation $i"
  end
  gif(anim, joinpath(dir, "evo.gif"), fps = 2)
end
