import unittest
from ..app import forex_io

class TestReadData(unittest.TestCase):

  def setUp(self):
    pass

  def test_read_train_data_minute_rows(self):
    print "Ovo je proba!"
    data=forex_io.read_train_data_minute_rows('eurusd_alpari', '2011.01.01', '2012.01.01')
    print "There are %d rows in data."%len(data)
    assert(len(data)==368617)

  def test_read_train_data(self):
    forex_io.read_train_data('eurusd_alpari', '2011.01.01', '2011.01.05', 5, 3)
    
if __name__=='__main__':
  unittest.main()
