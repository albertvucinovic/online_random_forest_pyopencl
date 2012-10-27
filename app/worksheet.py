import action_server.core.send_message_to_server as send_message


def read_train_data(currency, since, upto, minutes_before, minutes_after):
  return {
    'algo':{
      'method':'read',
      'exec':'from  ..app import forex_io',#import sys\nsys.path.append(\'..\')\nimport forex_io\n',
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

def send_action_with_name_helper(action, action_name, priority, redo):
  send_message.send_twidler_message("(%d,(%s,'%s',%d))"%(priority, action, action_name, redo))
  print "Sending action of priority %d, with name %s!"%(priority+1, action_name)


currency_pairs=['eurusd_alpari', 'gbpjpy_alpari']

priority=0

