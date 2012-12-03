from decision_tree import DecisionTree

from gini_opencl import OpenCLGiniCalculator

import numpy

from utils import *

from sklearn.mixture import GMM

class RegressionTreeSecretOpenCL(DecisionTree):
  
  opencl_calc=OpenCLGiniCalculator(class_type='float')
  def _find_best_split(self):
    print "OpenCL Regression split"
    #X=[0.9,1.,1.9,2.,2.1,1.1]
    X=[prediction for (feature_value,prediction) in enumerate(self._first_feature())]
    gmm=GMM(n_components=2, covariance_type='spherical', init_params='wc', n_iter=10)
    gmm.fit(X)
    classes=numpy.array(gmm.predict(X)).astype(numpy.float32)
    
    y_dim=self._seen_samples()
    x_dim=self.number_of_decision_functions

    A=numpy.empty((y_dim,x_dim)).astype(numpy.float32)
    for i, feature in enumerate(self.randomly_selected_features):
      for j, (feature_value, prediction) in enumerate(self.samples[feature]):
        A[j,i]=feature_value

    gini_matrix=RegressionTreeSecretOpenCL.opencl_calc.opencl_gini_matrix(A, classes)
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
      self.left=RegressionTreeSecretOpenCL(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'mean':best_split['left'].mean(),
          'variance':best_split['left'].var(),
          'num_samples':len(best_split['left'])
        }
      )
      self.right=RegressionTreeSecretOpenCL(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'mean':best_split['right'].mean(),
          'variance':best_split['right'].var(),
          'num_samples':len(best_split['right'])
        }
      )
      #collect garbage
      self.samples={}



  
