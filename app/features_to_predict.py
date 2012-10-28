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

  def feature_names(generic_name, minutes_after):
    names=[]
    for i in range(1, minutes_after+1):
      si=str(i)
      names.append(generic_name+si)
    return names


  def transform(self, features):
    pass


