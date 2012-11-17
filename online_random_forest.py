import random
import numpy 
from sklearn.mixture import GMM


class DecisionTreeNode:
  def __init__(self, 
      number_of_features,
      number_of_decision_functions=10,
      min_samples_to_split=100,
      predict_without_samples={
        'mean':0.0,
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
    self.randomly_selected_decision_functions=set([])
    self._randomly_select_decision_functions()
    self.criterion=None

  def _randomly_select_decision_functions(self):
      self._randomly_select_features()

  def seen_samples(self):
    return len(self.first_feature())

  def first_feature(self):
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

  def is_leaf(self):
    return self.left==None and self.right==None

  def update(self, x, y):
    self.update_statistics(x,y)
    if self.is_leaf():
      if self.seen_samples()>self.min_samples_to_split:
        self.find_and_apply_best_split()
    if not self.is_leaf():
      if self.criterion(x):
        self.right.update(x,y)
      else:
        self.left.update(x,y)

  def update_statistics(self, x, y):
      if self.is_leaf():#we stop collecting statistical info after the split
        for feature in self.randomly_selected_features:
          self.samples[feature].append((x[feature], y))
      

  def find_and_apply_best_split(self):
    pass

  def update_out_of_bag_error(self, x, y):
    #TODO:Estimate out of bag error
    pass

  def predict(self, x):
    """
      When the node is created, it gets only mean and sample count from the samples
      that the node "inherited" from its parent. 
      When the node has "seen" less than need to split samples, it takes "inhereted" samples to 
      compensate (actually it takes the mean of inhereted samples multiplied by the number of
      samples needed) when predicting.
    """
    if self.is_leaf():
      N=self.seen_samples()
      if N>0:
        how_many_needed_for_split=max(0,self.min_samples_to_split-N)
        how_many_inherited=min(how_many_needed_for_split, self.predict_without_samples['num_samples'])
        total=float(how_many_inherited+N)
        return (
          N/total*numpy.array(self.first_feature()).mean()+
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
      tolerance=1e-5):
    self.number_of_trees=number_of_trees
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.tolerance=tolerance
    self.trees=map(lambda x:DecisionTree(
      number_of_features,
      number_of_decision_functions_at_node
      ), range(number_of_trees))

  def update(self, x, y):
    for tree in self.trees:
      k=numpy.random.poisson()
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
    return sum(predictions)/len(predictions)
      

class OnlineRandomForestClassifier:
  pass

