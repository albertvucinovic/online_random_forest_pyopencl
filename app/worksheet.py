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


send_message.run()

