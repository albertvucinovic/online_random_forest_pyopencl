import numpy

#Takes a dictionary of values
def argmax(d):
    max_class=0
    max_count=0
    total_count=0
    for key, value in d.iteritems():
      total_count+=value
      if value>max_count:
        max_count=value
        max_class=key
    #returning the chosen class and the confidence
    if total_count==0:
      return (max_class, 0.)
    else:
      return (max_class, max_count/float(total_count))

#returns the element that shows up maximum number of times
def predict_max(a):
   return argmax(count_dict(a))

def count_dict(a):
  d={}
  for x in a:
    if x in d:
      d[x]+=1
    else:
      d[x]=1
  return d
      
def mean_square_error(x):
    xnp=numpy.array(x)
    xnp=xnp-xnp.mean()
    #return (xnp*xnp.T).mean()
    return abs(xnp).mean()
     
def gini(x):
  d={}
  for y in x:
    d[y]=0
  for y in x:
    d[y]+=1
  total=float(len(x))
  to_sqare=[]
  for key, value in d.iteritems():
    to_sqare.append(value/total)
  to_square=numpy.array(to_sqare)
  return 1-(to_square*to_square.T).sum()

    

def parallel_update(s, tree,x,y):
  s.update_tree(tree,x,y)
 
