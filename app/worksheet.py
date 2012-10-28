import action_server.core.send_message_to_server as send_message


def read_train_data_action(currency, since, upto, minutes_before, minutes_after):
  return {
    'algo':{
      'method':'read',
      'exec':'import app.forex_io',#import sys\nsys.path.append(\'..\')\nimport forex_io\n',
      'read method':'app.forex_io.read_train_data',
      'method args':{
        'currency':currency,
        'since':since,
        'upto':upto,
        'minutes_before':minutes_before,
        'minutes_after':minutes_after
      }
    }
  }

def send_action_with_name_helper(action, action_name, priority, redo):
  send_message.send_twidler_message("(%d,(%s,'%s',%d))"%(priority, action, action_name, redo))
  print "Sending action of priority %d, with name %s!"%(priority+1, action_name)


currency_pairs=['eurusd_alpari', 'gbpjpy_alpari']


since='2012.05.01'
upto='2012.06.01'
minutes_before=360
minutes_after=180
priority=0

for currency in currency_pairs:
  action=read_train_data_action(currency, since, upto, minutes_before, minutes_after)
  action_name='minute_data_360_180_2012_05_'+currency
  send_action_with_name_helper(action, action_name, priority, 0)
  priority+=1


def calculate_action_for_features(execute, algo_class, algo_args, features):
  return {
    'algo':{
      'method':'calculate',
      'exec':execute,
      'algo class': algo_class,
      'algo args': algo_args
    },
    'features':features,
    'partition':'lambda f:range(numpy.size(f))'
  }
def transform_features_action(algo_hash, features):
  return {
    'algo': {
      'method':'transform',
      'algo hash': algo_hash
    },
    'features':features,
    'partition':'lambda f:range(numpy.size(f))'
  }

def features_names(minutes_before, minutes_after):
  names=[]
  total=minutes_before+minutes_after
  map(lambda x:names.append(x), ['year', 'month', 'day', 'weekday', 'hour', 'minute'])
  for i in range(total):
    si=str(i-minutes_before+1)
    map(lambda x:names.append(x), ['open'+si, 'high'+si, 'low'+si, 'close'+si, 'volume'+si])
  return names

def calculate_features_on_all_sets(execute, algo_class, algo_args, action_name_short, redo):
  global priority
  for currency in currency_pairs:
    features=[]
    names="lambda f:"+str(features_names(minutes_before, minutes_after))
    features.append(('minute_data_360_180_2012_05_'+currency, names, names))
    calculate_action_name='calculate_'+action_name_short+'_minute_data_360_180_2012_05_'
    action=calculate_action_for_features(execute, algo_class, algo_args, features)
    action_name=calculate_action_name+currency
    send_action_with_name_helper(action,action_name,priority,redo)
    priority+=1
  for currency in currency_pairs:
    features=[]
    names="lambda f:"+str(features_names(minutes_before, minutes_after))
    features.append(('minute_data_360_180_2012_05_'+currency, names, names))
    transform_action_name='transform_'+action_name_short+'_minute_data_360_180_2012_05_'
    action_name=transform_action_name+currency
    action=transform_features_action(action_name, features)
    send_action_with_name_helper(action,action_name,priority,redo)
    priority+=1


#Features that are going to be predicted
#calculate_features_on_all_sets(
#  'import app.features_to_predict',
#  'app.features_to_predict.LowHighVolumeFeaturesToPredict',
#  {
#    'minutes_before':minutes_before,
#    'minutes_after':minutes_after
#  },
#  'low_high_volume_features_to_predict_fib_times_154',
#  0)

send_message.run()

