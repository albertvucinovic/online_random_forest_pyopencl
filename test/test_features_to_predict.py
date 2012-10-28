import unittest
import app.forex_io as forex_io
from app.features_to_predict import LowHighVolumeFeaturesToPredict
import numpy

class TestFeaturesToPredict(unittest.TestCase):
  def setUp(self):
    self.ftp=LowHighVolumeFeaturesToPredict(minutes_before=10,minutes_after=10,intervals=[1,2,3,5])

  def test_feature_names(self):
    a= self.ftp.feature_names('volume', 5)
    print a
    assert(a==['volume1', 'volume2', 'volume3', 'volume4', 'volume5'])
  
  def test_transform(self):
    data=forex_io.read_train_data('eurusd_alpari', '2012.05.01', '2012.05.05', 10,10)
    print data[[range(2)]][self.ftp.feature_names('volume', 5)]
    transformed_data=self.ftp.transform(data)
    print transformed_data
    assert transformed_data['volume_int_1'][0]==16
    print transformed_data['volume_int_1'][0]
    assert transformed_data['volume_int_2'][0]==36
    assert transformed_data['volume_int_3'][0]==42
    print data[[range(3)]][self.ftp.feature_names('high',5)]
    assert transformed_data['high_int_5'][1]==1.32413
    assert transformed_data['high_int_3'][1]==1.32398
    print data[[range(3)]][self.ftp.feature_names('low',5)]
    assert transformed_data['low_int_5'][1]==1.32388
    assert transformed_data['low_int_2'][1]==1.32391



if __name__=='__main__':
  unittest.main()
