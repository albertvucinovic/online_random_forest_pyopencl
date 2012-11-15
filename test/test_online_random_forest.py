import online_random_forest.online_random_forest as orf
import unittest

class TestOnlineRandomForest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_create_DecisionTree(self):
    dt=orf.DecisionTree(3)


if __name__=='__main__':
  unittest.main()
