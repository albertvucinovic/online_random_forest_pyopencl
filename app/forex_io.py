import psycopg2

def read_train_data(currency, since, upto, minutes_before, minutes_after):
  conn=psycopg2.connect("dbname=forex user=albert")
  cur=conn.cursor()
  cur.execute("select * from %s where date>=%s and date<%s"%(currency, since, upto))
  data=cur.fetchall()
