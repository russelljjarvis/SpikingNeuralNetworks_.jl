import neo
import sciunit
from networkunit import models, tests, scores, plots, capabilities
import sciunit
import neo
import numpy as np
from networkunit import tests, scores, models
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
import julia
from julia.api import JuliaInfo
from julia import Base
import julia

try:
    from julia import Julia
    from julia.api import Julia
except:
    julia.install()
    from julia import Julia
    from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main
import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt

from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_rasterplot import rasterplot

from networkunit import models, tests, scores, plots, capabilities
from networkunit.plots import sample_histogram
import neo
import numpy as np
import quantities as qt
import sciunit

jl.using("JLD")
exec_string = """
filename = string("../JLD/PopulationScatter_adexp.jld")
if isfile(filename)
    opt_vec = load(filename, "opt_vec")
    extremum_param = load(filename,"extremum_param")
    vmgtt = load(filename, "vmgtt")
    vmgtv = load(filename, "vmgtv")
    ground_spikes = load(filename,"ground_spikes")
    opt_spikes = load(filename, "opt_spikes")
end
"""
Main.eval(exec_string)
exec_string2="""
using SignalAnalysis
using Plots
s_a = signal(opt_vec, 1) # 1KHz
println(vmgtt[2])
s_b = signal(vmgtv, vmgtt[2]) # 100 Hz
p = plot(s_a) # just plot it
p2 = plot!(p, s_b)
savefig(p2, "example.png")

"""
Main.eval(exec_string2)

#ts=target.tspkt.spikes
import pyspike as spk
ALLEN_DURATION = 2000 * qt.ms
ALLEN_DELAY = 1000 * qt.ms
t = ALLEN_DURATION+ALLEN_DELAY


opt_vec = Main.opt_vec
extremum_param = Main.extremum_param
vmgtt = Main.vmgtt
vmgtv = Main.vmgtv


vm1 = neo.AnalogSignal(opt_vec, units=qt.mV, sampling_period=1*qt.ms)
vm2 = neo.AnalogSignal(vmgtv, units=qt.mV, sampling_period=Main.vmgtt[1]*qt.s)


plt.plot(vm1.times,vm1)
plt.plot(vm2.times,vm2)
plt.savefig("single_cell_opt_out.png")



print("data loaded")
#opt_vec = load(filename, "opt_vec")
#opt_time=vmgtt[2]:vmgtt[2]:length(opt_vec)/vmgtt[2]
#p1=
#for t in opt_time
#    print(t)
#end



class single_polychrony_data(models.spiketrain_data, ProducesSpikeTrains):
    """
    Re-define NetworkUnits
    """
    def __init__(self, t_stop, spike_times):
        self = self
        self.t_start = 0
        self.t_stop = t_stop
        self.nbr_neurons = 1
        self.spike_times = spike_times
        spiketrains = [[]] * self.nbr_neurons
        n = 0
        #for i, st in enumerate(self.spike_times):
        spiketrain = neo.core.SpikeTrain(
            self.spike_times, units="ms", t_start=self.t_start, t_stop=self.t_stop)
        self.spiketrain = spiketrain
        self.spiketrains = spiketrain

    def produce_spiketrains(self, **kwargs):
        """
        overwrites function in capability class ProduceSpiketrains
        """
        return self.spiketrain

    def show_rasterplot(self, **kwargs):
        return rasterplot(self.spiketrain, **kwargs)


class FR_test_class(sciunit.TestM2M, tests.firing_rate_test):
    score_type = scores.effect_size


class LV_test_class(sciunit.TestM2M, tests.isi_variation_test):
    score_type = scores.effect_size
    params = {"variation_measure": "lv"}


class isi_ttest_class(sciunit.TestM2M, tests.isi_variation_test):
    score_type = scores.students_t
    params = {"variation_measure": "isi"}

    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score



class fr_ttest_class(sciunit.TestM2M, tests.firing_rate_test):
    score_type = scores.students_t
    params = {"equal_var": False}  # True: Student's t-test; False: Welch's t-test

    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score


class fr_effect_class(sciunit.TestM2M, tests.firing_rate_test):
    score_type = scores.effect_size


ts=target.tspkt.spikes
import pyspike as spk
ALLEN_DURATION = 2000 * qt.ms
ALLEN_DELAY = 1000 * qt.ms
t = ALLEN_DURATION+ALLEN_DELAY

#ground_spikes = load(ground_spikes,"ground_spikes")
#opt_spikes =
#ts = spk.SpikeTrain(ts, edges=(0,t), is_sorted=True)
ts = Main.ground_spikes
ts = spk.SpikeTrain(ts, edges=(0,t), is_sorted=True)
os = Main.opt_spikes#threshold_detection(opt.vM)

#os = threshold_detection(opt.vM)
os = spk.SpikeTrain(os, edges=(0,t), is_sorted=True)

#opt.get_initial_vm()

m1 = single_polychrony_data(t, 1, 1,[ts])  # ,name='ground truth')
m2 = single_polychrony_data(t, 1, 1, [os])  # ,name='optimized candidate')


FR_test = FR_test_class()
LV_test = LV_test_class()
isi_ttest = isi_ttest_class()


fr_effect = fr_effect_class()


fr_ttest = fr_ttest_class(equal_var=True)
pred0 = fr_ttest.generate_prediction(m1)
pred1 = fr_ttest.generate_prediction(m2)
score = fr_ttest.compute_score(pred0, pred1)


pred0 = isi_ttest.generate_prediction(m1)
pred1 = isi_ttest.generate_prediction(m2)
isi_score = isi_ttest.compute_score(pred0, pred1)
print(isi_score)


pred0 = LV_test.generate_prediction(m1)
pred1 = LV_test.generate_prediction(m2)
LV_score = LV_test.compute_score(pred0, pred1)
print(LV_score)
