import unittest
from online_random_forest.gini_opencl import kernel, kernel_arguments
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
    self.ctx=cl.create_some_context()
    for dev in self.ctx.devices:
      assert dev.local_mem_size>0
    self.queue=cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    if "NVIDIA" in self.queue.device.vendor:
      options="-cl-mad-enable -cl-fast-relaxed-math"
    else:
      options=""
    kernel_params=kernel_arguments
    self.height=height
    self.width=width
    kernel_params['class_type']='int'
    kernel_params['num_features']=self.width
    kernel_params['num_samples']=self.height
    kernel_params['prime']= 31#4294967291
    print kernel_params
    self.prg=cl.Program(self.ctx, kernel%kernel_params).build(options=options) 
    self.kernel=self.prg.gini
    self.h_a=A
    self.h_a_gini=numpy.zeros((self.height, self.width)).astype(numpy.float32)
    self.classes=classes



  def test_normal_gini(self):
    self.gini_matrix()
    print "Normal Gini Matrix:", self.h_a_gini

  def test_opencl_gini(self):
    print "OpenCL Gini Matrix:", self.opencl_gini_matrix()

  def opencl_gini_matrix(self):
    debug_length=4+self.height*3+self.width
    #getting data to the device
    mf=cl.mem_flags
    t1=time()
    d_a_buf=cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=self.h_a)
    d_classes_buf=cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=self.classes)
    print self.h_a_gini.nbytes
    push_time=time()-t1
    #calculating one gini matrix
    d_debug_buf=cl.Buffer(self.ctx, mf.WRITE_ONLY, size=debug_length*4)
    d_gini_buf=cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.h_a_gini.nbytes)
    d_return_A_buf=cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.h_a.nbytes)
    event=self.kernel(self.queue, self.h_a.shape[::-1], (1,self.height), d_a_buf, d_classes_buf, d_gini_buf, d_debug_buf, d_return_A_buf)
    event.wait()
    gpu_time=time()-t1

    #getting data from device
    h_a_gini=numpy.empty((self.height, self.width)).astype(numpy.float32)
    h_debug=numpy.empty(debug_length).astype(numpy.int32)
    h_return_a=numpy.empty((self.height, self.width)).astype(numpy.float32)
    t1=time()
    cl.enqueue_copy(self.queue, h_a_gini, d_gini_buf)
    cl.enqueue_copy(self.queue, h_debug, d_debug_buf)
    cl.enqueue_copy(self.queue, h_return_a, d_return_A_buf)
    pull_time=time()-t1
    print "Push time, gpu time, pull time:", push_time, gpu_time, pull_time
    return h_a_gini
    #print "OpenCL Gini Matrix:", h_a_gini
    #for x in range(h_a_gini.shape[0]):
    #  for y in range(h_a_gini.shape[1]):
    #    print x,y,h_a_gini[x,y]
    #print h_debug
    #print self.classes
    #print "h_return_a",h_return_a


  def gini_matrix(self):
    for feature in range(self.width):
      for sample in range(self.height):
        threshold=self.h_a[sample,feature]
        myerror=gini(self.classes)
        left=[self.classes[i] for i in range(self.height) if self.h_a[i, feature]<=threshold]
        right=[self.classes[i] for i in range(self.height) if self.h_a[i, feature]>threshold]
        if feature==0 and sample==0:
          print "A[0,0]", self.h_a[0,0]
          print "A[max,max]", self.h_a[self.height-1, self.width-1]
          print "Left:",left
          print "Right:", right
        left_error=gini(left)
        right_error=gini(right)
        self.h_a_gini[sample, feature]=myerror-1/float(self.height)*(len(left)*left_error+len(right)*right_error)



  def tearDown(self):
    pass


if __name__=='__main__':
  unittest.main()

