"""A relative difference between prediction and observation.
The absolute value of the difference between the prediction and the
observation is divided by a reference value with the same units. This
reference scale should be chosen for each test such that normalization
produces directly comparable scores across tests. For example, if 5 volts
represents a medium size difference for TestA, and 10 seconds represents a
medium size difference for TestB, then 5 volts and 10 seconds should be
used for this reference scale in TestA and TestB, respectively. The
attribute `scale` can be passed to the compute method or set for the whole
class in advance. Otherwise, a scale of 1 (in the units of the
observation and prediction) will be used.
"""

#    _allowed_types = (float,)

#    _description = ('The relative difference between the prediction and the observation')
_best = 0.0  # A RelativeDifferenceScore of 0.0 is best
_worst = inf
scale = None

function _check_score(self, score):
        if score < 0.0:
            raise errors.InvalidScoreError(("RelativeDifferenceScore was initialized with "
                                            "a score of %f, but a RelativeDifferenceScore "
                                            "must be non-negative.") % score)
        end
end
function compute(cls, observation: Union[dict, float, int, pq.Quantity],
                 prediction: Union[dict, float, int, pq.Quantity],
                 key=None,
                 scale: Union[float, int, pq.Quantity, None] = None) -> 'RelativeDifferenceScore':
    """Compute the relative difference between the observation and a prediction.
    Returns:
        RelativeDifferenceScore: A relative difference between an observation and a prediction.
    """
    assert isinstance(observation, (dict, float, int, pq.Quantity))
    assert isinstance(prediction, (dict, float, int, pq.Quantity))

    obs, pred = cls.extract_means_or_values(observation, prediction,
                                            key=key)


    scale = scale or cls.scale or (obs/float(obs))
    assert type(obs) is type(scale)
    assert type(obs) is type(pred)

    if isinstance(obs, pq.Quantity):
        assert obs.units == pred.units, \
            "Prediction must have the same units as the observation"
        assert obs.units == scale.units, \
            "RelativeDifferenceScore.Scale must have the same units as the observation"
        pred = pred.rescale(obs.units)

    assert scale > 0, \
        "RelativeDifferenceScore.scale must be positive (not %g)" % scale
    value = np.abs(pred - obs) / scale
    value = utils.assert_dimensionless(value)
    return RelativeDifferenceScore(value)
end

function norm_score(self) -> float:
        """Return 1.0 for a ratio of 0.0, falling to 0.0 for extremely large values.
        Returns:
            float: The value of the norm score.
        """
        x = self.score
        return 1 / (1+x)
end
function __str__(self):
        return 'Relative Difference = %.2f' % self.score
end
