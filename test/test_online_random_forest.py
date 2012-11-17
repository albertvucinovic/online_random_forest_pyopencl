import online_random_forest.online_random_forest as orf
import unittest
from sklearn.mixture import GMM

class TestOnlineRandomForest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_create_DecisionTree(self):
    dt=orf.DecisionTree(10)

  def test_GMM(Self):
    X=[0.9,1.,1.9,2.,2.1,1.1]
    gmm=GMM(n_components=2, covariance_type='spherical', init_params='wc', n_iter=20)
    gmm.fit(X)
    y_train_predict=gmm.predict(X)
    assert list(y_train_predict)==[1,1,0,0,0,1] or list(y_train_predict)==[0,0,1,1,1,0]
    assert gmm.means_.mean()>1.45 and gmm.means_.mean()<1.55



if __name__=='__main__':
  unittest.main()
