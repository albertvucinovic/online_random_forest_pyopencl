import unittest
from online_random_forest.gini_opencl import OpenCLGiniCalculator
import pyopencl as cl
import numpy
from time import time

from online_random_forest.online_random_forest import gini

height=8
width=16
A=numpy.random.rand(height, width).astype(numpy.float32)
classes=numpy.array(map(lambda x:numpy.random.randint(5), range(height))).astype(numpy.int32)

class TestOpenCLGini(unittest.TestCase):
  def setUp(self):
    self.opencl_calc=OpenCLGiniCalculator(height, width)
    self.classes=classes



  def test_normal_gini(self):
    print "Normal Gini Matrix:", self.gini_matrix(A, classes)

  def test_opencl_gini(self):
    print "OpenCL Gini Matrix:", self.opencl_gini_matrix()

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

