from classification_tree import ClassificationTree

from gini_opencl import OpenCLGiniCalculator

import numpy
from utils import *

class ClassificationTreeOpenCL(ClassificationTree):
  opencl_calc=OpenCLGiniCalculator(class_type='float')
  def _find_best_split(self):
    print "OpenCLSplit"
    y_dim=self._seen_samples()
    x_dim=self.number_of_decision_functions

    A=numpy.empty((y_dim,x_dim)).astype(numpy.float32)
    for i, feature in enumerate(self.randomly_selected_features):
      for j, (feature_value, prediction) in enumerate(self.samples[feature]):
        A[j,i]=feature_value
        
    classes=numpy.empty(y_dim).astype(numpy.float32)
    for j, (feature_value, prediction) in enumerate(self.samples[self.randomly_selected_features[0]]):
      classes[j]=float(prediction)

    gini_matrix=ClassificationTreeOpenCL.opencl_calc.opencl_gini_matrix(A, classes)
    argmax=gini_matrix.argmax()

    y_max=argmax/x_dim
    x_max=argmax%x_dim

    feature_value=A[y_max,x_max]
    feature=self.randomly_selected_features[x_max]
    best_split={
          'left': numpy.array([x[1] for x in self.samples[feature] if x[0]<=feature_value]),
          'right': numpy.array([x[1] for x in self.samples[feature] if x[0]>feature_value]),
          'threshold':feature_value,
          'feature':feature
        }

    best_split_score=gini_matrix[y_max, x_max]

    return (best_split, best_split_score)

  def _find_and_apply_best_split(self):
    (best_split, best_split_score)=self._find_best_split()
    if best_split_score>0:
      print self._first_feature(), gini(self._first_feature())
      print best_split['left'], gini(best_split['left'])
      print best_split['right'], gini(best_split['right'])
      self.criterion=lambda x:x[best_split['feature']]>best_split['threshold']
      self.left=ClassificationTreeOpenCL(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'count_dict':count_dict(best_split['left']),
        }
      )
      self.right=ClassificationTreeOpenCL(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'count_dict':count_dict(best_split['right']),
        }
      )
      #collect garbage
      self.samples={}






