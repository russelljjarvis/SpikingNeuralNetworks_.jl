using JLD
using Plots
#import Pkg; Pkg.add("SignalAnalysis")
using SignalAnalysis
using Plots

#unicodeplots()
filename = string("../JLD/PopulationScatter_adexp.jld")#, py"target_num_spikes")#,py"specimen_id)

if isfile(filename)
    #trace = load(filename, "trace")
    opt_vec = load(filename, "opt_vec")
    extremum_param = load(filename,"extremum_param")
    vmgtt = load(filename, "vmgtt")
    vmgtv = load(filename, "vmgtv")
end

println("data loaded")
#opt_vec = load(filename, "opt_vec")
#opt_time=vmgtt[2]:vmgtt[2]:length(opt_vec)/vmgtt[2]
#p1=
#for t in opt_time
#    print(t)
#end


function vecplot(v)
    #v = SNN.getrecord(p, sym)
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
