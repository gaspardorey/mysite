import pandas as pd
import numpy as np
import math


#def promotions():
# Load the data

# Sales -- dsProgReports:
dsSales = pd.read_csv('C:/Users/gadorey/projects/myPandas/sales_smaller.csv')
# Customers -- dsDistrict:
dsCustomers = pd.read_csv('C:/Users/gadorey/projects/myPandas/customers_small.csv')
# Promotions -- dsClassSize:
dsPromotions = pd.read_csv('C:/Users/gadorey/projects/myPandas/promotions_small.csv')
# Products -- dsAttendEnroll:
dsProducts = pd.read_csv('C:/Users/gadorey/projects/myPandas/products_small.csv') #[:-2] # last two rows are bad
# ???? -- dsSATs -- unused for now:
# dsSATs = pd.read_csv('C:/Users/gadorey/projects/myPandas/SAT__College_Board__2010_School_Level_Results.csv') # Dependent

# >>> pdPredict.dsSATs.info()


# ESTRATÉGIA:
#Target Variable and Joining Strategy
# We are going to build a dataset to predict Sales for each Product (identified by SKU).
# After digging around in Excel (or just taking my word for it) we identify the following join
# strategy (using SQL-esque pseudocode):

# dsSales join dsProducts on dsSales['SKU'] = dsProducts['SKU']
# join dsCustomers on dsSales['CustomerID'] = dsCustomers['CustomerID']
# join dsPromotions on dsSales['SKU'] = dsPromotions['SKU'] AND date(dsSales['InvoiceDate']) > dsPromotions['start_date'] AND date(dsSales['InvoiceDate']) < dsPromotions['end_date']

#dsCustomers.info()
#dsSales.info()

PrimNormalized = pd.DataFrame(data=[dsSales['SKU'][:5], dsPromotions['SKU'][:5], dsProducts['id'][:5]])

# Strip the first two characters off the DBNs so we can join to School Code
#dsProgReports.DBN = dsProgReports.DBN.map(lambda x: x[2:])
#dsSATs.DBN=dsSATs.DBN.map(lambda x: x[2:])


#We can now see the keys match:
# use slicing instead of "take" ie [:5] instead of "take(range(5))"
#data=pd.DataFrame(data=[dsProgReports['DBN'][:5], dsSATs['DBN'][:5], dsClassSize['SCHOOL CODE'][:5]])
#print('Data Joined: ')
#print(data)
#print('******')


# Show the key mismatchs [13]
# For variety's sake, using slicing ([:3]) sintax instead of .take()
data= pd.DataFrame(data=[dsSales['CustomerID'][:3], dsCustomers['custId'][:3]])
print('Customer Keys Mismatch')
print(data)
print('******')

# Extract well-formed district key values [14]
# Note the astype(int) at the end of these lines to coerce the column to a 
# numeric type:
import re
#dsSales['CustomerID']=dsDSales['CustomerID'].map(lambda x: re.match(r'([A-Za-z]*\s)([0-9]*)',x).group(2)).astype(int)
#dsAttendEnroll.District = dsAttendEnroll.District.map(lambda x: x[-2:]).astype(int)

# We can now see the keys match
new_data=pd.DataFrame(data=[dsSales['CustomerID'][:3], dsCustomers['custId'][:3]])

print('Customer Keys Match = ')
print(new_data)
print('******')

# Reindexing
dsSales = dsSales.set_index('SKU')
dsProducts = dsProducts.set_index('id')
dsCustomers = dsCustomers.set_index('custId')
dsPromotions = dsPromotions.set_index('SKU')

#print('dsProgReports = ')
#print(dsProgReports)
#print('******')
#print('dsSATs = ')
#print(dsSATs)
#print('******')

# We can see the bad values
# dsSATs['Critical Reading Mean'].take(range(5))
new_dsSales = dsSales['Quantity'][:5]
print('Here is an example of the "s" values in dsSATs = ')
print(new_dsSales)
print('******')

# Now we filter it out
# We create a boolean vector mask. Open question as to whether this is 
# semantically ideal....
#mask = dsSales['Number of Test Takers'].map(lambda x: x != 's')
#dsSATs = dsSATs[mask]

# Cast fields to integers. Ideally we should not need to be this explicit.
dsSales['Quantity'] = dsSales['Quantity'].astype(int)
dsSales['InvoiceDate'] = pd.to_datetime(dsSales['InvoiceDate'],format="%d/%m/%Y %H:%M")
dsPromotions['START_DATE'] = pd.to_datetime(dsPromotions['START_DATE'])
dsPromotions['END_DATE'] = pd.to_datetime(dsPromotions['END_DATE'])

#print('Mask = ')
#print(mask)
#print('*********')
#print('Old dsSATs = ')
#print(new_dsSATs)
#print('*********')
#print('New dsSATs = ')
#print(dsSATs['Critical Reading Mean'][:5])
#print('*********')
#print('New dsSATs complete = ')
#print(dsSATs[:5])
#print('*********')


# The shape of the data:
#print('dsClassSize.columns = ',dsClassSize.columns)
#print('*************')
#print('dsClassSize.take([0,1,10]).values = ',dsClassSize.take([0,1,10]).values)
#print('*************')

# Extracting the Pupil-Teacher ratio:

# 	1) a) Take the Column
dsEventID = dsPromotions.filter(['EVENT_ID'])
print('Old dsEventID = ')
print(dsEventID[:15])
print('*************')

# 		b) And filter out blank rows
mask = dsEventID['EVENT_ID'].map(lambda x: x > 0)
dsEventID = dsEventID[mask]
print('mask = ')
print(mask[:15])
print('*************')
print('New dsEventID = ')
print(dsEventID[:15])
print('*************')


##########################
# CHECK THIS HERE:
##########################

# 		c) Then drop the original dataset
#dsClassSize = dsClassSize.drop('SCHOOLWIDE PUPIL-TEACHER RATIO', axis=1)
#print('New dsClassSize = ')
#print(dsClassSize[:15])
#print('*************')

############################
############################

# Changin "Yes" for "1" on "Non-Target" Column:

# 	1) a) Take the Column
dsNonTarget = dsPromotions.filter(['NON_TARGET'])
print('Old dsNonTarget = ')
print(dsNonTarget[:15])
print('*************')
'''
# 		b) And filter out blank rows
mask = dsNonTarget['NON_TARGET'].map(lambda x: 1 if x == 'YES' else 0)
dsNonTarget = dsNonTarget[mask]
print('mask = ')
print(mask[:15])
print('*************')
print('New dsNonTarget = ')
print(dsNonTarget[:15])
print('*************')
'''

#	2) Drop non-numeric fields
dsPromotions=dsPromotions.drop(['EVENT_DSC','BRAND','CAMPAIGN_TYPE','TV','RADIO','UPLIFT_BASE','SYSTEM_EVENT_ID_GROUP','NUM_CLIENTS','CLIENTS_GROUP'], axis=1)
print('New non-numeric dsPromotionsNew 1 = ')
print(dsPromotions[:15])
print('*************')

dsSales=dsSales.drop(['InvoiceNo','Description','CustomerID'], axis=1)
print('New non-numeric dsSalesNew 1 = ')
print(dsSales[:15])
print('*************')


'''

#	3) Build features from dsPromotions
#		In this case, we'll take the max, min and mean
#		Semantically equivalent to select min(*), max(*) from dsClassSize
#		group by SCHOOL NAME
#		Note that SCHOOL NAME is not referenced explicitly below because
#		it is the index of the dtaframe
grouped = dsPromotions.groupby(level='SKU')
dsPromotions = grouped.aggregate(np.max).\
	join(grouped.aggregate(np.min),lsuffix=".max").\
	join(grouped.aggregate(np.mean),lsuffix=".min",rsuffix=".mean")
	#join(grouped.aggregate(np.mean),lsuffix=".min",rsuffix=".mean").\
	#join(dsNonTarget)
	
print('Grouped dsPromotionsNewG columns = ')
print(dsPromotions.columns)
print('*************')			
print('Grouped dsPromotionsNewG = ')
print(dsPromotions[:5])
print('*************')	

'''

# One final thing before we join - dsProgReports contains distinct 
# rows for separate grade level blocks within one school. For instance 
# one school (one DBN) might have two rows: one for middle school and 
# one for high school. We'll just drop everything that isn't high school.
# And finally we can join our data. Note these are inner joins, so 
# district data get joined to each school in that district.

#mask = dsProgReports['SCHOOL LEVEL*'].map(lambda x: x == 'High School')
#dsProgReports = dsProgReports[mask]

############

# dsSales = dsSales.set_index('SKU')
# dsProducts = dsProducts.set_index('id')
# dsCustomers = dsCustomers.set_index('custId')
# dsPromotions = dsPromotions.set_index('SKU')

# dsSales join dsProducts on dsSales['SKU'] = dsProducts['id']
# join dsCustomers on dsSales['CustomerID'] = dsCustomers['CustomerID']
# join dsPromotions on dsSales['SKU'] = dsPromotions['SKU'] AND date(dsSales['InvoiceDate']) > dsPromotions['start_date'] AND date(dsSales['InvoiceDate']) < dsPromotions['end_date']

print("final = dsSales.join(dsPromotions):")
final = dsSales.join(dsPromotions)
print(final[:5])
print("*********")

'''
TRY THIS LATER, AS NEXT LINE IS NOT WORKING VERY WELL:

print("final = dsSales.join(dsPromotions).join(dsProducts):")
final = dsSales.join(dsPromotions).merge(dsProducts, left_on='SKU', right_index=True)
print(final[:5])
print("*********")

print("final = dsSales.join(dsPromotions).join(dsProducts).merge(dsCustomers, left_on='CustomerID', right_index=True):")
final = dsSales.join(dsPromotions).join(dsProducts).merge(dsCustomers, left_on='CustomerID', right_index=True)
print(final[:5])
print("*********")

#final = dsSales.join(dsPromotions).join(dsProducts).merge(dsCustomers, left_on='CustomerID', right_index=True)


print('Final = ')
print(final[:5])
print('*************')	
'''

# (Even More) Additional Cleanup
# We should be in a position to build a predictive model for our target variables right away but
# unfortunately there is still messy data floating around in the dataframe that machine learning
# algorithms will choke on. A pure feature matrix should have only numeric features, but we can
# see that isn't the case. However for many of these columns, the right approach is obvious once
# we've dug in.

'''
CHANGE NON_TARGET TO INTEGER - CHECK ABOVE:

print('final.columns = ')
print(final.columns)
print('*************')

print('final[NON_TARGET] = ')
print(final['NON_TARGET'])
print('*************')
'''

print('final.dtypes = ')
final.dtypes[final.dtypes.map(lambda x: x=='object')]
print(final.dtypes[final.dtypes.map(lambda x: x=='object')])
print('*************')	



#Just drop string columns. [23]
#In theory we could build features out of some of these, but it is impractical here.

print('final with dropped = ')
final = final.drop(['Store','NON_TARGET'],axis=1)
#final = final.drop(['InvoiceNo','Description','Store','Store','NON_TARGET.max','NON_TARGET'],axis=1)
#final = final.drop(['CustomerID'],axis=1) # Try this.
print(final[:5])
print('*************')	

'''
# Remove % signs and convert to float
final['YTD % Attendance (Avg)'] = final['YTD % Attendance (Avg)'].map(lambda x: x.replace("%","")).astype(float)
print('final no % = ')
print(final['YTD % Attendance (Avg)'][:5])
print('*************')	
'''

'''
# The last few columns we still have to deal with
final.dtypes[final.dtypes.map(lambda x: x=='object')]
print('final.dtypes = ')
print(final['YTD % Attendance (Avg)'][:5])
print('*************')	
'''

# Categorical Variables [24]
# We can see above that the remaining non-numeric field are grades . Intuitively, they might be
# important so we don't want to drop them, but in order to get a pure feature matrix we need
# numeric values. The approach we'll use here is to explode these into multiple boolean
# columns. Some machine learning libraries effectively do this for you under the covers, but
# when the cardinality of the categorical variable is relatively low, it's nice to be explicit about it.

#gradeCols = ['DISCOUNT']

print('final.columns = ')
print(final.columns)
print('*************')

print('salesCols = ')
salesCols = ['Quantity']
print(salesCols)
print('*************')

print('final[salesCols] = ')
print(final[salesCols][:5])
print('*************')


#[nan, A, B, C, D, F]
#grades=np.unique(final[gradeCols].values)
#grades1=pd.unique(final['DISCOUNT'].values)
#grades2=pd.unique(final['2009-2010 ENVIRONMENT GRADE'].values)
#grades3=pd.unique(final['2009-2010 PERFORMANCE GRADE'].values)
#grades4=pd.unique(final['2009-2010 PROGRESS GRADE'].values)
#grades5=pd.unique(final['2008-09 PROGRESS REPORT GRADE'].values)

#for g in grades1:
#	final = final.join(pd.Series(data=final['DISCOUNT'].map(lambda x: 1 if x is g else 0), name = "DISCOUNT_is_" + str(g)))

#for g in grades2:
#	final = final.join(pd.Series(data=final['2009-2010 ENVIRONMENT GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 ENVIRONMENT GRADE_is_" + str(g)))

#for g in grades3:
#	final = final.join(pd.Series(data=final['2009-2010 PERFORMANCE GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 PERFORMANCE GRADE_is_" + str(g)))

#for g in grades4:
#	final = final.join(pd.Series(data=final['2009-2010 PROGRESS GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 PROGRESS GRADE_is_" + str(g)))

#for g in grades5:
#	final = final.join(pd.Series(data=final['2008-09 PROGRESS REPORT GRADE'].map(lambda x: 1 if x is g else 0), name = "2008-09 PROGRESS REPORT GRADE_is_" + str(g)))

#print('pd.Series = ')
#print(pd.Series(data=final['2009-2010 OVERALL GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 OVERALL GRADE_is_" + str(g)))
#print('*************')
	
#for c in gradeCols:
#	for g in grades:
#		final = final.join(pd.Series(data=final[c].map(lambda x: 1 if x is g else 0), name = c + "_is_" + str(g)))

#final = final.drop(gradeCols, axis=1)


print('final = ')
print(final[:5])
print('*************')


#That's it!
#We now have a feature matrix that's trivial to use with any number of machine learning
#algorithms. Feel free to stop here and run the two lines of code below to get nice CSV files
#written to disk that you can easily use in Excel, Tableau, etc. Or run the larger block of code to
#see how easy it is to build a random forest model against this data, and look at which variables
#are most important.

# Uncomment to generate CSV files:
#final.to_csv('C:/Users/gadorey/projects/myPandas/results/Promo_full.csv')
final.drop(['Quantity'],axis=1).to_csv('C:/Users/gadorey/projects/myPandas/results/Promo_train.csv')
final.filter(['Quantity']).to_csv('C:/Users/gadorey/projects/myPandas/results/Promo_target.csv')

######################

#from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

target=final.filter(['Quantity'])
print('target = ')
print(target)
print('*************')

# We drop all three dependent variables because we dont want them used when trying to make a prediction

#train = final.drop(['Quantity'],axis=1)
train = final.filter(['InvoiceDate','UnitPrice','EVENT_ID','START_DATE','END_DATE','MARKET_GROUP_ID','ORG_GROUP_ID','DISCOUNT'])

#print('train = ')
#print(train['High School'][:5])
#print('*************')
print('target columns = ')
print(target.columns)
print('*************')	

print('train columns = ')
print(train.columns)
print('*************')	
print('train = ')
print(train[:5])
print('*************')

###  TESTING TRAIN FOR NaN  --> TO REMOVE LATER
#InvoiceDate
#START_DATE
#END_DATE

train['InvoiceDate']=pd.to_datetime(train['InvoiceDate'],format="%Y-%m-%d %H:%M:%S")
print(train['InvoiceDate'][:5])
train['START_DATE']=pd.to_datetime(train['START_DATE'],format="%Y-%m-%d")
print(train['START_DATE'][:5])
train['END_DATE']=pd.to_datetime(train['END_DATE'],format="%Y-%m-%d")
print(train['END_DATE'][:5])

'''
print('train_is_nan - per column = ')
train_is_nan=np.any(np.isnan(train['InvoiceDate']))
print(train_is_nan)
'''
train = final.filter(['UnitPrice','EVENT_ID','MARKET_GROUP_ID','ORG_GROUP_ID','DISCOUNT'])

print('target_is_nan= ')
target_is_nan=np.any(np.isnan(target))
print(target_is_nan)

print('target_is_infinite= ')
target_is_infinite=np.all(np.isfinite(target))
print(target_is_infinite)

print('train_is_nan= ')
train_is_nan=np.any(np.isnan(train))
print(train_is_nan)

print('target_is_infinite= ')
train_is_infinite=np.all(np.isfinite(train))
print(train_is_infinite)

print('*************')	
#target_inf=np.isinf(target)
#train_inf=np.isinf(train)

#target_inf.to_csv('C:/Users/gadorey/projects/myPandas/results/target_inf.csv')
#train_inf.to_csv('C:/Users/gadorey/projects/myPandas/results/train_inf.csv')

#target_nan=np.isnan(target)
#train_nan=np.isnan(train)

#target_nan.to_csv('C:/Users/gadorey/projects/myPandas/results/target_nan.csv')
#train_nan.to_csv('C:/Users/gadorey/projects/myPandas/results/train_nan.csv')

target2=target
where_are_NaNs_tg = np.isnan(target2)
target2[where_are_NaNs_tg] = 0
target2.to_csv('C:/Users/gadorey/projects/myPandas/results/Promo_target2.csv')

train2=train
where_are_NaNs = np.isnan(train2)
train2[where_are_NaNs] = 0
train2.to_csv('C:/Users/gadorey/projects/myPandas/results/Promo_train2.csv')

del final
del dsSales
del dsProducts
del dsPromotions
del train
del target
del PrimNormalized
del data
del dsCustomers
del dsEventID
del dsNonTarget
del mask
del new_data
del new_dsSales
del salesCols

print(dir())

# model=RandomForestRegressor(n_estimators=100, n_jobs=-1, compute_importances=True)
model=RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(train2,target2)

predictions=np.array(model.predict(train2))
rmse=math.sqrt(np.mean((np.array(target2.values)-predictions)**2))
imp=sorted(zip(train2.columns,model.feature_importances_),key=lambda tup: tup[1],reverse=True)

print('RMSE: '+str(rmse))
print('10 Most important variables: '+str(imp[:10]))
print('******************')

return render(request, 'polls/predict.html', {'rmse': rmse})



'''

# Logistic Regression Python

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# load the iris / final datasets
dataset = datasets.load_iris()
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(dataset.data, dataset.target)
print('Iris Model = ')
print(model)
print('***********')

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print('Iris - Fit of the Model - Classification = ')
print(metrics.classification_report(expected, predicted))
print('***********')
print('Iris - Fit of the Model - Confusion = ')
print(metrics.confusion_matrix(expected, predicted))



# Logistic Regression
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

'''
