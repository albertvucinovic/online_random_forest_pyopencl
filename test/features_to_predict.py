import unittest
import app.forex_io
from app.features_to_predict import LowHighVolumeFeaturesToPredict

class TestFeaturesToPredict(unittest.TestCase):
  def setUp(self):
    self.ftp=LowHighVolumeFeaturesToPredict(360,180)

  def test_feature_names(self):
    print self.ftp.feature_names('volume', 5)
  
  def test_transform(self):
    pass


if __name__=='__main__':
  unittest.main()
