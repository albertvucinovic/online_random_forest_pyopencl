import random
import numpy 
from sklearn.mixture import GMM

total_splits=0

class DecisionTreeNode:
  def __init__(self, 
      number_of_features,
      number_of_decision_functions=10,
      min_samples_to_split=20,
      predict_without_samples={
        'mean':2.0,
        'variance':1.0,
        'num_samples':0
      }
    ):
    #Constants
    self.number_of_features=number_of_features
    self.number_of_decision_functions=number_of_decision_functions
    self.min_samples_to_split=min_samples_to_split
    self.predict_without_samples=predict_without_samples
    #Dynamic
    self.left=None #False branch
    self.right=None #True branch
    self.randomly_selected_decision_functions={}
    self._randomly_select_decision_functions()
    self.criterion=None


  def _randomly_select_decision_functions(self):
      self._randomly_select_features()

  def _seen_samples(self):
    return len(self._first_feature())

  def _first_feature(self):
    #x[1] is the predicted value for the sample
    return map(lambda x:x[1],self.samples[self.randomly_selected_features[0]])


  def _randomly_select_features(self):
    if self.number_of_features<self.number_of_decision_functions:
      raise Exception('Cant have more randomly selected features than features')
    self.randomly_selected_features=set([])
    while len(self.randomly_selected_features)<self.number_of_decision_functions:
      self.randomly_selected_features.add(random.randint(0,self.number_of_decision_functions-1))
    #I turn it into a list for convenience
    self.randomly_selected_features=list(self.randomly_selected_features)
    self.samples={}
    for feature in self.randomly_selected_features:
      self.samples[feature]=[]#initialize storage for statistics

  def _is_leaf(self):
    return self.criterion==None

  def update(self, x, y):
    if self._is_leaf():
      N=self._seen_samples()
      #Statistics for maximum 2*self.min_samples_to_split are collected
      #after that, we never split the node, and stop updating the statistics
      #if N<=self.min_samples_to_split:
      self._update_statistics(x,y)
      if N>self.min_samples_to_split:
        self._find_and_apply_best_split()
    if not self._is_leaf():
      if self.criterion(x):
        self.right.update(x,y)
      else:
        self.left.update(x,y)

  def _update_statistics(self, x, y):
    for feature in self.randomly_selected_features:
      self.samples[feature].append((x[feature], y))
      
  def _mean_square_error(self, x):
    xnp=numpy.array(x)
    xnp=xnp-xnp.mean()
    return (xnp*xnp.T).mean()

  def _my_mean_square_error(self):
    return self._mean_square_error(self._first_feature())

  def _calculate_split_score(self, split):
      #if the split is any good, this number should be greater than 0
      left_error=self._mean_square_error(split['left'])
      right_error=self._mean_square_error(split['right'])
      myerror=self._my_mean_square_error()
      score =myerror-max(left_error,right_error)
      #print myerror, left_error, right_error, score
      return score

  def _find_best_split(self):
    best_split=None
    best_split_score=0
    for feature in self.randomly_selected_features:
      for (feature_value, prediction) in self.samples[feature]:
        split={
          'left': numpy.array([x[1] for x in self.samples[feature] if x[0]<=feature_value]),
          'right': numpy.array([x[1] for x in self.samples[feature] if x[0]>feature_value]),
          'threshold':feature_value,
          'feature':feature
        }
        split_score=self._calculate_split_score(split)
        if(split_score>best_split_score):
          best_split=split
          best_split_score=split_score

    return (best_split, best_split_score)

  def _find_and_apply_best_split(self):
    global total_splits
    (best_split, best_split_score)=self._find_best_split()
    if best_split_score>0:
      total_splits+=1
      print "                      ", total_splits, len(best_split['left']), len(best_split['right'])
      print self._first_feature(), self._my_mean_square_error()
      print best_split['left'], self._mean_square_error(best_split['left'])
      print best_split['right'], self._mean_square_error(best_split['right'])
      self.criterion=lambda x:x[best_split['feature']]>best_split['threshold']
      self.left=DecisionTreeNode(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'mean':best_split['left'].mean(),
          'variance':best_split['left'].var(),
          'num_samples':len(best_split['left'])
        }
      )
      self.right=DecisionTreeNode(
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
      
  def update_out_of_bag_error(self, x, y):
    #TODO:Estimate out of bag error
    pass

  def predict(self, x):
    """
      When the node is created, it gets only mean and sample count from the samples
      that the node "inherited" from its parent. 
      When the node has "seen" less than min_to_split samples, it takes "inhereted" samples to 
      compensate (actually it takes the mean of inhereted samples multiplied by the number of
      samples needed) when predicting.
    """
    if self._is_leaf():
      N=self._seen_samples()
      if N>0:
        how_many_needed_for_split=max(0,self.min_samples_to_split-N)
        how_many_inherited=min(how_many_needed_for_split, self.predict_without_samples['num_samples'])
        total=float(how_many_inherited+N)
        #print self._first_feature()
        return (
          N/total*numpy.array(self._first_feature()).mean()+
          how_many_inherited/total*self.predict_without_samples['mean'])
      else:
        return self.predict_without_samples['mean']
    else:
      if self.criterion(x):
        return self.right.predict(x)
      else:
        return self.left.predict(x)
      

class DecisionTree(DecisionTreeNode):
  pass

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
      tolerance=1e-5):
    self.number_of_trees=number_of_trees
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.tolerance=tolerance
    self.trees=map(lambda x:DecisionTree(
      number_of_features,
      number_of_decision_functions_at_node,
      min_samples_to_split=number_of_samples_to_split,
      ), range(number_of_trees))

  def update(self, x, y):
    for tree in self.trees:
      #k=numpy.random.poisson()#this is with resampling
      k=numpy.random.randint(2)#don't want to resample
      if k>0:
        for i in range(k):
          tree.update(x,y)
      else:
        #If k==0, then the sample is not learned by the tree
        tree.update_out_of_bag_error(x,y)
   

  def predict(self, x):
    predictions=[]
    for tree in self.trees:
      predictions.append(tree.predict(x))
    predictions=numpy.array(predictions)
    return (predictions.mean(), predictions.var())
      

class OnlineRandomForestClassifier:
  pass

