import psycopg2

def read_train_data(currency, since, upto, minutes_before, minutes_after):
  data=read_train_data_minute_rows(currency, since, upto)

def read_train_data_minute_rows(currency, since, upto):
  conn=psycopg2.connect("dbname=forex user=albert")
  cur=conn.cursor()
  cur.execute("select * from %s where date>='%s' and date<'%s'"%(currency, since, upto))
  data =cur.fetchall()
  cur.close()
  conn.close()
  return data

