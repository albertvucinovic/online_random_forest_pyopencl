import psycopg2
import datetime
import numpy

def create_train_row(aligned_rows,minutes_before, minutes_after, zero_index):
  row=[]
  total=minutes_before+minutes_after
  #input features
  for i in range(minutes_before):
    for j in range(2,7):#open, high, low, close, volume
      row.append(aligned_rows[zero_index+i][1][j])
  #time related basic features
  dt=aligned_rows[zero_index+minutes_before][0]
  map(lambda x:row.append(x), [dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute])
  #features for prediction/goal setting
  for i in range(minutes_before, total):
    for j in range(2,7):#open, high, low, close, volume
      row.append(aligned_rows[zero_index+i][1][j])
  return row

def row_dtypes(minutes_before, minutes_after):
  row_dtypes=[]
  total=minutes_before+minutes_after
  for i in range(minutes_before):
    si=str(i-minutes_before+1)
    map(lambda x:row_dtypes.append((x,numpy.float)), ['open'+si, 'high'+si, 'low'+si, 'close'+si, 'volume'+si])
  map(lambda x:row_dtypes.append((x,numpy.int)),['year', 'month', 'day', 'weekday', 'hour', 'minute'])
  for i in range(minutes_before, total):
    si=str(i-minutes_before+1)
    map(lambda x:row_dtypes.append((x,numpy.float)), ['open'+si, 'high'+si, 'low'+si, 'close'+si, 'volume'+si])
  return row_dtypes


def read_train_data(currency, since, upto, minutes_before, minutes_after):
  data=read_train_data_minute_rows(currency, since, upto)
  #put data into a hash table indexed by time
  time_hashed_rows={}
  for row in data:
    dt=datetime.datetime.strptime(row[0]+' '+str(row[1]), "%Y.%m.%d %H:%M:%S")
    time_hashed_rows[dt]=row
  print "There are %d records in the hash"%len(time_hashed_rows)

  #first take minutes_before+minutes_after consecutive (have data for almost all minutes) rows
  since_t=datetime.datetime.strptime(since, "%Y.%m.%d")
  upto_t=datetime.datetime.strptime(upto, "%Y.%m.%d")
  td=datetime.timedelta(seconds=60)
  current_time=since_t
  aligned_rows=[]
  while current_time<upto_t:
    if current_time in time_hashed_rows:
      aligned_rows.append([current_time, time_hashed_rows[current_time]])
    else:
      aligned_rows.append([current_time, None])
    current_time+=td
  #searching for appropriate candidates
  training_rows=[]
  pos=0
  number_of_missing=0
  total=minutes_before+minutes_after
  while pos<len(aligned_rows):
    if aligned_rows[pos][1]==None:
      number_of_missing+=1
    if pos>=total:
      if aligned_rows[pos-total][1]==None:
        number_of_missing-=1
    if pos>=total-1:#we have at least total samples (indexes start with 0)
      #if number_of_missing<total/10:#we have at least 90% of data
      if number_of_missing==0: #We have all the data in the total interval
        row=create_train_row(aligned_rows, minutes_before, minutes_after, pos-total+1)
        training_rows.append(row)
    pos+=1
  row_dtypes=row_dtypes(minutes_before, minutes_after)
  print row_dtypes
  return numpy.array(training_rows, dtype=row_dtypes)
        
def read_train_data_minute_rows(currency, since, upto):
  conn=psycopg2.connect("dbname=forex user=albert")
  cur=conn.cursor()
  cur.execute("select * from %s where date>='%s' and date<'%s'"%(currency, since, upto))
  data =cur.fetchall()
  cur.close()
  conn.close()
  return data

