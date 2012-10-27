def read_train_data(currency, since, upto, minutes_before, minutes_after):
  return {
    'algo':{
      'method':'read',
      'exec':'import sys\nsys.path.append(\'..\')\nimport forex_io\n',
      'read method':'forex_io.read_train_data',
      'method args':{
        'currency':currency,
        'since':since,
        'upto':upto,
        'minutes_before':minutes_before,
        'minutes_after':minutes_after
      }
    }
  }
