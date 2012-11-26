import unittest
from online_random_forest.gini_opencl import OpenCLGiniCalculator
import pyopencl as cl
import numpy
from time import time

from online_random_forest.online_random_forest import gini

height=16
width=1024
A=numpy.random.rand(height, width).astype(numpy.float32)
classes=numpy.array(map(lambda x:numpy.random.randint(5), range(height))).astype(numpy.int32)

class TestOpenCLGini(unittest.TestCase):
  def setUp(self):
    self.opencl_calc=OpenCLGiniCalculator(height, width)
    self.classes=classes



  def test_opencl_gini(self):
    t=time()
    gini_matrix=self.gini_matrix(A, classes)
    time_spent=time()-t
    print "Normal Gini Matrix:", gini_matrix
    self.normal_time=time_spent
    print "Took seconds:", time_spent
    t=time()
    opencl_gini_matrix=self.opencl_gini_matrix()
    time_spent=time()-t
    print "OpenCL Gini Matrix:", opencl_gini_matrix
    self.gini_time=time_spent
    print "Took seconds:", time_spent
    print "Speedup is:", self.normal_time/time_spent
    assert numpy.allclose(gini_matrix, opencl_gini_matrix, atol=1e-6)



  def opencl_gini_matrix(self):
    h_a_gini=self.opencl_calc.opencl_gini_matrix(A, classes)
    return h_a_gini
    
  def gini_matrix(self, A, classes):
    h_a_gini=numpy.zeros((height, width)).astype(numpy.float32)
    for feature in range(width):
      for sample in range(height):
        threshold=A[sample,feature]
        myerror=gini(classes)
        left=[classes[i] for i in range(height) if A[i, feature]<=threshold]
        right=[classes[i] for i in range(height) if A[i, feature]>threshold]
        left_error=gini(left)
        right_error=gini(right)
        h_a_gini[sample, feature]=myerror-1/float(height)*(len(left)*left_error+len(right)*right_error)
    return h_a_gini



  def tearDown(self):
    pass


if __name__=='__main__':
  unittest.main()

