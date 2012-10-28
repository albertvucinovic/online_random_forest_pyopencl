import numpy
class LowHighVolumeFeaturesToPredict:
  """
  A class that creates features that need to be predicted.
  It generates low, high, volume features for 1,2,3,5,8,13,21,34,55,89,154 minutes intervals.
  """
  def __init__(self, minutes_before, minutes_after, intervals=[1,2,3,5,8,13,21,34,55,89,154]):
    self.minutes_before=minutes_before
    self.minutes_after=minutes_after
    self.intervals=intervals

  def fit(self, features):
    pass

  def feature_names(self,generic_name, minutes_after):
    names=[]
    for i in range(1, minutes_after+1):
      si=str(i)
      names.append(generic_name+si)
    return names


  def transform(self, features):
    dtypes=[]
    calculated_features=[]
    for interval in self.intervals:
      volumes_interval=numpy.sum(map(lambda x: list(x),features[self.feature_names('volume', interval)]),1)
      calculated_features.append(volumes_interval)
      dtypes.append(('volume_int_'+str(interval), numpy.float))

      high_interval=numpy.max(map(lambda x: list(x), features[self.feature_names('high', interval)]),1)
      calculated_features.append(high_interval)
      dtypes.append(('high_int_'+str(interval), numpy.float))

      low_interval=numpy.min(map(lambda x: list(x), features[self.feature_names('low', interval)]),1)
      calculated_features.append(low_interval)
      dtypes.append(('low_int_'+str(interval), numpy.float))

    calculated_features=numpy.array(calculated_features).T
    return numpy.array(map(lambda x: tuple(x), calculated_features), dtype=dtypes)


