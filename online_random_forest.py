import random
import numpy


#from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool as Pool

from utils import *
from decision_tree import DecisionTree
from classification_tree import ClassificationTree
from classification_tree_opencl import ClassificationTreeOpenCL
from regression_tree_secret_opencl import RegressionTreeSecretOpenCL


class OnlineRandomForestRegressor:
  """
  This class implements Online Random Forest, and is inpired by articles:
    On-line Random Forest, Amir Saffari et al.
    SECRET: A Scalable Linear Regression Tree Algorithm, Dobra, Gehrke
  """
  def __init__(self, 
      number_of_features, 
      number_of_trees=100, 
      number_of_decision_functions_at_node=10, 
      number_of_samples_to_split=10,
      tolerance=1e-5,
      tree_class=DecisionTree):
    self.number_of_trees=number_of_trees
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.tolerance=tolerance
    self.trees=map(lambda x:tree_class(
      number_of_features,
      number_of_decision_functions_at_node,
      min_samples_to_split=number_of_samples_to_split,
      ), range(number_of_trees))

    self.pool=Pool(2)

  def update_tree(self, tree, x, y):
      #k=numpy.random.poisson()#this is with resampling
      k=numpy.random.randint(2)#don't want to resample
      if k>0:
        for i in range(k):
          tree.update(x,y)
      else:
        #If k==0, then the sample is not learned by the tree
        tree.update_out_of_bag_error(x,y)

  def list_trees(self):
    for tree in self.trees:
      yield tree
    
  def update(self, x, y):

    self.pool.map(lambda x:parallel_update(*x), [(self, tree, x, y) for tree in self.trees])

    #doesn't work
    #jobs=Parallel(n_jobs=1, pre_dispatch='2', verbose=1)(
    #  delayed(parallel_update)(self,tree,x,y) for tree in self.list_trees())

    #Single threaded
    #for tree in self.trees:
    #  self.update_tree(tree, x, y)
      
         

  def predict(self, x):
    predictions=[]
    for tree in self.trees:
      predictions.append(tree.predict(x))
    predictions=numpy.array(predictions)
    return (predictions.mean(), predictions.var())
      

class OnlineRandomForestClassifier(OnlineRandomForestRegressor):
  def __init__(self,
    number_of_features,
    number_of_trees=100,
    number_of_decision_functions_at_node=10,
    number_of_samples_to_split=2,
    tree_class=ClassificationTree
  ):
    self.number_of_features=number_of_features
    self.number_of_trees=number_of_trees
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.trees=map(lambda x:tree_class(
      number_of_features,
      number_of_decision_functions_at_node,
      min_samples_to_split=number_of_samples_to_split),
      range(number_of_trees))
    self.pool=Pool(2)

  def predict(self, x):
    predictions=[tree.predict(x) for tree in self.trees]
    return predict_max(predictions)

class OnlineRandomForestClassifierOpenCLSplit(OnlineRandomForestClassifier):
  def __init__(self,
    number_of_features,
    number_of_trees=100,
    number_of_decision_functions_at_node=10,
    number_of_samples_to_split=2,
  ):
    OnlineRandomForestClassifier.__init__(
      self,
      number_of_features,
      number_of_trees,
      number_of_decision_functions_at_node,
      number_of_samples_to_split,
      tree_class=ClassificationTreeOpenCL)



class OnlineRandomForestRegressorSecretOpenCL(OnlineRandomForestRegressor):
  def __init__(self,
    number_of_features,
    number_of_trees=100,
    number_of_decision_functions_at_node=10,
    number_of_samples_to_split=2,
  ):
    OnlineRandomForestRegressor.__init__(
      self,
      number_of_features,
      number_of_trees,
      number_of_decision_functions_at_node,
      number_of_samples_to_split,
      tree_class=RegressionTreeSecretOpenCL)

 


