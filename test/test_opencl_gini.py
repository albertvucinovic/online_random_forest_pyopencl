import unittest
from online_random_forest.gini_opencl import kernel, kernel_arguments
import pyopencl as cl
import numpy
from time import time

from online_random_forest.online_random_forest import gini

class TestOpenCLGini(unittest.TestCase):
  def setUp(self):
    self.ctx=cl.create_some_context()
    for dev in self.ctx.devices:
      assert dev.local_mem_size>0
    self.queue=cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    if "NVIDIA" in self.queue.device.vendor:
      options=""#-cl-mad-enable -cl-fast-relaxed-math"
    else:
      options=""
    kernel_params=kernel_arguments
    self.height=32
    self.width=8192
    kernel_params['class_type']='int'
    kernel_params['num_features']=self.width
    kernel_params['num_samples']=self.height
    kernel_params['prime']= 31#4294967291
    print kernel_params
    self.prg=cl.Program(self.ctx, kernel%kernel_params).build(options=options) 
    self.kernel=self.prg.gini
    self.h_a=numpy.random.rand(self.height, self.width).astype(numpy.float32)
    self.h_a_gini=numpy.zeros((self.height, self.width)).astype(numpy.float32)
    self.classes=numpy.array(map(lambda x:numpy.random.randint(5), range(self.height))).astype(numpy.int32)



  def test_normal_gini(self):
    self.gini_matrix()
    print self.h_a_gini

  def test_opencl_gini(self):
    self.opencl_gini_matrix()

  def opencl_gini_matrix(self):
    debug_length=4+self.height*3
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
    event=self.kernel(self.queue, self.h_a.shape[::-1], (1,self.height), d_a_buf, d_classes_buf, d_gini_buf, d_debug_buf)
    event.wait()
    gpu_time=time()-t1

    #getting data from device
    h_a_gini=numpy.empty((self.height, self.width)).astype(numpy.float32)
    h_debug=numpy.empty(debug_length).astype(numpy.int32)
    t1=time()
    cl.enqueue_copy(self.queue, h_a_gini, d_gini_buf)
    cl.enqueue_copy(self.queue, h_debug, d_debug_buf)
    pull_time=time()-t1
    print h_a_gini
    #for x in range(h_a_gini.shape[0]):
    #  for y in range(h_a_gini.shape[1]):
    #    print x,y,h_a_gini[x,y]
    print push_time, gpu_time, pull_time
    print h_debug
    print self.classes


  def gini_matrix(self):
    for feature in range(self.width):
      for sample in range(self.height):
        threshold=self.h_a[sample,feature]
        myerror=gini(self.classes)
        left=[self.classes[i] for i in range(self.height) if self.h_a[i, feature]<=threshold]
        right=[self.classes[i] for i in range(self.height) if self.h_a[i, feature]>threshold]
        #print "Left:",left
        #print "Right:", right
        left_error=gini(left)
        right_error=gini(right)
        self.h_a_gini[sample, feature]=myerror-1/float(self.height)*(len(left)*left_error+len(right)*right_error)



  def tearDown(self):
    pass


if __name__=='__main__':
  unittest.main()

