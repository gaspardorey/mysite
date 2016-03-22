import plotly.tools as tls
#tls.embed('https://plot.ly/~chris/7365')
tls.set_credentials_file(username='gaspardorey',api_key='fobabamwnw')

import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
import sqlite3 # to delete database

import plotly.plotly as py # interactive graphing
from plotly.graph_objs import Bar, Scatter, Marker, Layout

#display(pd.read_csv('311_100M.csv', nrows=2).head()) # Display only two rows
#display(pd.read_csv('311_100M.csv', nrows=2).tail())

original=pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/holt_sales.csv')
originalhead=pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/holt_sales.csv', nrows=2, low_memory=False).head() # Display only two rows
#originaltail=pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/311_100M.csv', nrows=2).tail()

print(originalhead)
#print(originaltail)

#!wc -l < 311_100M.csv
#!wc -l < holt_sales.csv
#print(len(original.index))

# Initializes database with filename 311_8M.db in current directory:
#disk_engine = create_engine('sqlite:///311_100M.db')
disk_engine = create_engine('sqlite:///holt_sales.db')

####
#### Delete all previous records
conn = sqlite3.connect('holt_sales.db')
print("Connection OK")

df = pd.read_sql_query('SELECT * FROM data', disk_engine)
print("df = ")
print(df[:5])

c = conn.cursor()
mydata = c.execute("DELETE FROM data")
conn.commit()
c.close
print("All previous records deleted.")
#####

start = dt.datetime.now()
chunksize = 20 #20000 for 311_100M
j = 0
index_start = 1

for df in pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/holt_sales.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):
	df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns
	df['Period'] = pd.to_datetime(df['Period']) # Convert to datetimes
    #df['ClosedDate'] = pd.to_datetime(df['ClosedDate'])
	df.index += index_start
	
	'''
	# In case you need to Remove the un-interesting columns:
    columns = ['Agency', 'CreatedDate', 'ClosedDate', 'ComplaintType', 'Descriptor',
               'CreatedDate', 'ClosedDate', 'TimeToCompletion',
               'City']
			   
	for c in df.columns:
        if c not in columns:
            df = df.drop(c, axis=1)
	'''
	
	j+=1
	print ('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j*chunksize))
	df.to_sql('data', disk_engine, if_exists='append')
	index_start = df.index[-1] + 1
	
# Preview the table
df = pd.read_sql_query('SELECT * FROM data LIMIT 3', disk_engine)
print(df)

# Select just a couple of columns
#df = pd.read_sql_query('SELECT Agency, Descriptor FROM data LIMIT 3', disk_engine)
df = pd.read_sql_query('SELECT Sales FROM data LIMIT 3', disk_engine)
print(df.head())

# LIMIT the number of rows that are retrieved
#df = pd.read_sql_query('SELECT ComplaintType, Descriptor, Agency '
df = pd.read_sql_query('SELECT Sales '
                       'FROM data '
                       'LIMIT 10', disk_engine)

print(df)

# Filter rows with WHERE
df = pd.read_sql_query('SELECT Sales, Period '
                       'FROM data '
                       'WHERE Sales = "500" '
                       'LIMIT 10', disk_engine)
print(df)

# Filter multiple values in a column with WHERE and IN
df = pd.read_sql_query('SELECT Period, Sales '
                       'FROM data '
                       'WHERE Sales IN ("500", "600")'
                       'LIMIT 10', disk_engine)
					   
print(df)

# Find the unique values in a column with DISTINCT and ORDER BY
df = pd.read_sql_query('SELECT DISTINCT Sales FROM data ORDER BY Sales', disk_engine)
print(df.head())

# Query value counts with COUNT(*) and GROUP BY
df = pd.read_sql_query('SELECT Sales, COUNT(*) as `num_sales`'
                       'FROM data '
                       'GROUP BY Sales ', disk_engine)
print(df.head())

# Order the results with ORDER and -
df = pd.read_sql_query('SELECT Sales, Period, COUNT(*) as `num_sales`'
                       'FROM data '
                       'GROUP BY Sales '
                       'ORDER BY -num_sales', disk_engine)

					   
#py.iplot([Bar(x=df.Sales, y=df.num_sales)], filename='holt_sales/most common sales by sales')
#py.iplot([line(x=df.Period, y=df.Sales)], filename='holt_sales/sales per Period')
py.iplot([Scatter(x=df.Period, y=df.Sales)], mode='lines+markers', name="'Sales'", hoverinfo='name',line=dict(shape='linear'))


