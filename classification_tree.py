from decision_tree_node import DecisionTreeNode

class ClassificationTree(DecisionTreeNode):
  def __init__(self, 
      number_of_features,
      number_of_decision_functions=10,
      min_samples_to_split=20,
      predict_without_samples={
        'count_dict':{},
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
    self.predict_without_samples=predict_without_samples

  def _calculate_split_score(self, split):
    left_error=gini(split['left'])
    right_error=gini(split['right'])
    myerror=gini(self._first_feature())
    total=float(len(self._first_feature()))
    score=myerror-1/total*(len(split['left'])*left_error+len(split['right'])*right_error)
    return score

  def _find_and_apply_best_split(self):
    (best_split, best_split_score)=self._find_best_split()
    if best_split_score>0:
      print self._first_feature(), gini(self._first_feature())
      print best_split['left'], gini(best_split['left'])
      print best_split['right'], gini(best_split['right'])
      self.criterion=lambda x:x[best_split['feature']]>best_split['threshold']
      self.left=ClassificationTree(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'count_dict':count_dict(best_split['left']),
        }
      )
      self.right=ClassificationTree(
        number_of_features=self.number_of_features,
        number_of_decision_functions=self.number_of_decision_functions,
        min_samples_to_split=self.min_samples_to_split,
        predict_without_samples={
          'count_dict':count_dict(best_split['right']),
        }
      )
      #collect garbage
      self.samples={}

  def predict(self, x):
    if self._is_leaf():
      d1=self.predict_without_samples['count_dict']
      d2=count_dict(self._first_feature())
      for key, value in d1.iteritems():
        if key in d2:
          d2[key]+=value
        else:
          d2[key]=value
      return argmax(d2)
    else:
      if self.criterion(x):
        return self.right.predict(x)
      else:
        return self.left.predict(x)
