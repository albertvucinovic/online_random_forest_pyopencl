class DecisionTreeNode:
  def __init__(self, number_of_decision_functions_at_node):
    self.is_leaf=True
    self.left=None #False branch
    self.right=None #True branch
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.decision_functions=[]

  def is_leaf(self):
    return self.left==None and self.right==None

class DecisionTree(DecisionTreeNode):
  pass

class DecisionFunction:
  def __init__(self, f):
    self.f=f

  def __call__(self, x):
    return f(x)

class OnlineRandomForestRegressor:
  def __init__(self, number_of_features, number_of_trees=100, number_of_decision_functions_at_node=10, tolerance=1e-5):
    self.number_of_trees=number_of_trees
    self.number_of_decision_functions_at_node=number_of_decision_functions_at_node
    self.tolerance=tolerance
    self.trees=map(lambda x:DecisionTree(number_of_decision_functions_at_node), range(number_of_trees))

  def update(self, x, y):
    for tree in self.trees:
      tree.update(x,y)
   

  def predict(self, x):
    predictions=[]
    for tree in self.trees:
      predictions.append(tree.predict(x))
    return sum(predictions)/len(predictions)
      

class OnlineRandomForestClassifier:
  pass

