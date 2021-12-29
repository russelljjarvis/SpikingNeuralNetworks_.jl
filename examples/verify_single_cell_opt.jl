using JLD
using Plots
using SignalAnalysis
using Plots

filename = string("../JLD/PopulationScatter_adexp.jld")#, py"target_num_spikes")#,py"specimen_id)

if isfile(filename)
    opt_vec = load(filename, "opt_vec")
    extremum_param = load(filename,"extremum_param")
    vmgtt = load(filename, "vmgtt")
    vmgtv = load(filename, "vmgtv")
end

println("data loaded")


function vecplot(v)
    y = hcat(v...)'
    x = 1:length(v)
    #t = [i/1000.0 for i in x]
    plot(x, y, leg = :none,
    xaxis=("t", extrema(x)),
    yaxis=(string("V_{m}"), extrema(y)))
end
p1 = vecplot(opt_vec)
p1|>display
p2 = vecplot(vmgtv[:])
p2|>display
s_a = signal(opt_vec, length(opt_vec)/last(vmgtt))
s_b = signal(vmgtv, length(vmgtt)/last(vmgtt))
p = plot(s_a)
p2 = plot!(p, s_b)
display(plot(s_a))
display(plot(s_b))
savefig(p2, "aligned_VM.png")
