import neo
import numpy as np
from networkunit import tests, scores, models
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
import matplotlib
import matplotlib.pyplot as plt

from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_rasterplot import rasterplot

from networkunit import models, tests, scores, plots, capabilities
from networkunit.plots import sample_histogram
import neo
import numpy as np
import quantities as qt
import sciunit
from networkunit import models, tests, scores, plots, capabilities
from elephant.spike_train_generation import threshold_detection


class single_polychrony_data(models.spiketrain_data, ProducesSpikeTrains):
    """
    Re-define NetworkUnits
    """

    def __init__(self, t_stop, 1 spike_times):
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

ts = spk.SpikeTrain(ts, edges=(0,t), is_sorted=True)
os = threshold_detection(opt.vM)
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
