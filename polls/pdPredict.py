x\import pandas as pd
import numpy as np
import math
import re
from sklearn.ensemble import RandomForestRegressor


# Load the data

dsProgReports = pd.read_csv('C:/Users/gadorey/projects/myPandas/School_Progress_Reports_-_All_Schools_-_2009-10.csv')
dsDistrict = pd.read_csv('C:/Users/gadorey/projects/myPandas/School_District_Breakdowns.csv')
dsClassSize = pd.read_csv('C:/Users/gadorey/projects/myPandas/2009-10_Class_Size_-_School-level_Detail.csv')
dsAttendEnroll = pd.read_csv('C:/Users/gadorey/projects/myPandas/School_Attendance_and_Enrollment_Statistics_by_District__2010-11_.csv')[:-2] # last two rows are bad
dsSATs = pd.read_csv('C:/Users/gadorey/projects/myPandas/SAT__College_Board__2010_School_Level_Results.csv') # Dependent

# >>> pdPredict.dsSATs.info()


# ESTRATÉGIA:
#Target Variable and Joining Strategy
# We are going to build a dataset to predict Critical Reading Mean, Mathematics Mean, and
# Writing Mean for each school (identified by DBN).
# After digging around in Excel (or just taking my word for it) we identify the following join
# strategy (using SQL-esque pseudocode):

# dsSATS join dsClassSize on dsSATs['DBN'] = dsClassSize['SCHOOL CODE']
# join dsProgReports on dsSATs['DBN'] = dsProgReports['DBN']
# join dsDistrct on dsProgReports['DISTRICT'] = dsDistrict['JURISDICTION NAME']
# join dsAttendEnroll on dsProgReports['DISTRICT'] = dsAttendEnroll['District']


PrimNormalized = pd.DataFrame(data=[dsProgReports['DBN'][:5], dsSATs['DBN'][:5], dsClassSize['SCHOOL CODE'][:5]])

# Strip the first two characters off the DBNs so we can join to School Code
dsProgReports.DBN = dsProgReports.DBN.map(lambda x: x[2:])
dsSATs.DBN=dsSATs.DBN.map(lambda x: x[2:])


#We can now see the keys match:
# use slicing instead of "take" ie [:5] instead of "take(range(5))"
data=pd.DataFrame(data=[dsProgReports['DBN'][:5], dsSATs['DBN'][:5], dsClassSize['SCHOOL CODE'][:5]])
#print('Data Joined: ')
#print(data)
#print('******')


# Show the key mismatchs [13]
# For variety's sake, using slicing ([:3]) sintax instead of .take()
data= pd.DataFrame(data=[dsProgReports['DISTRICT'][:3], dsDistrict['JURISDICTION NAME'][:3], dsAttendEnroll['District'][:3]])
#print('Keys Mismatch')
#print(data)
#print('******')

# Extract well-formed district key values [14]
# Note the astype(int) at the end of these lines to coerce the column to a 
# numeric type:

dsDistrict['JURISDICTION NAME']=dsDistrict['JURISDICTION NAME'].map(lambda x: re.match(r'([A-Za-z]*\s)([0-9]*)',x).group(2)).astype(int)
dsAttendEnroll.District = dsAttendEnroll.District.map(lambda x: x[-2:]).astype(int)

# We can now see the keys match
new_data=pd.DataFrame(data=[dsProgReports['DISTRICT'][:3],dsDistrict['JURISDICTION NAME'][:3],dsAttendEnroll['District'][:3]])

#print('Keys Match = ')
#print(new_data)
#print('******')

# Reindexing
dsProgReports = dsProgReports.set_index('DBN')
dsDistrict = dsDistrict.set_index('JURISDICTION NAME')
dsClassSize = dsClassSize.set_index('SCHOOL CODE')
dsAttendEnroll = dsAttendEnroll.set_index('District')
dsSATs = dsSATs.set_index('DBN')

#print('dsProgReports = ')
#print(dsProgReports)
#print('******')
#print('dsSATs = ')
#print(dsSATs)
#print('******')

# We can see the bad values
# dsSATs['Critical Reading Mean'].take(range(5))
new_dsSATs = dsSATs['Critical Reading Mean'][:5]
#print('Here is an example of the "s" values in dsSATs = ')
#print(new_dsSATs)
#print('******')

# Now we filter it out
# We create a boolean vector mask. Open question as to whether this is 
# semantically ideal....
mask = dsSATs['Number of Test Takers'].map(lambda x: x != 's')
dsSATs = dsSATs[mask]

# Cast fields to integers. Ideally we should not need to be this explicit.
dsSATs['Number of Test Takers'] = dsSATs['Number of Test Takers'].astype(int)
dsSATs['Critical Reading Mean'] = dsSATs['Critical Reading Mean'].astype(int)
dsSATs['Mathematics Mean'] = dsSATs['Mathematics Mean'].astype(int)
dsSATs['Writing Mean'] = dsSATs['Writing Mean'].astype(int)

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
dsPupilTeacher = dsClassSize.filter(['SCHOOLWIDE PUPIL-TEACHER RATIO'])
#print('Old dsPupilTeacher = ')
#print(dsPupilTeacher[:15])
#print('*************')

# 		b) And filter out blank rows
mask = dsPupilTeacher['SCHOOLWIDE PUPIL-TEACHER RATIO'].map(lambda x: x > 0)
dsPupilTeacher = dsPupilTeacher[mask]
#print('mask = ')
#print(mask[:15])
#print('*************')
#print('New dsPupilTeacher = ')
#print(dsPupilTeacher[:15])
#print('*************')

# 		c) Then drop the original dataset
dsClassSize = dsClassSize.drop('SCHOOLWIDE PUPIL-TEACHER RATIO', axis=1)
#print('New dsClassSize = ')
#print(dsClassSize[:15])
#print('*************')

#	2) Drop non-numeric fields
dsClassSize=dsClassSize.drop(['BORO','CSD','SCHOOL NAME','GRADE ','PROGRAM TYPE','CORE SUBJECT (MS CORE and 9-12 ONLY)','CORE COURSE (MS CORE and 9-12 ONLY)','SERVICE CATEGORY(K-9* ONLY)','DATA SOURCE'], axis=1)
#print('New non-numeric dsClassSize = ')
#print(dsClassSize[:15])
#print('*************')

#	3) Build features from dsClassSize
#		In this case, we'll take the max, min and mean
#		Semantically equivalent to select min(*), max(*) from dsClassSize
#		group by SCHOOL NAME
#		Note that SCHOOL NAME is not referenced explicitly below because
#		it is the index of the dtaframe
grouped = dsClassSize.groupby(level=0)
dsClassSize = grouped.aggregate(np.max).\
	join(grouped.aggregate(np.min),lsuffix=".max").\
	join(grouped.aggregate(np.mean),lsuffix=".min",rsuffix=".mean").\
	join(dsPupilTeacher)
	
#print('Grouped dsClassSize columns = ')
#print(dsClassSize.columns)
#print('*************')			
#print('Grouped dsClassSize = ')
#print(dsClassSize[:5])
#print('*************')	

# One final thing before we join - dsProgReports contains distinct 
# rows for separate grade level blocks within one school. For instance 
# one school (one DBN) might have two rows: one for middle school and 
# one for high school. We'll just drop everything that isn't high school.
# And finally we can join our data. Note these are inner joins, so 
# district data get joined to each school in that district.

mask = dsProgReports['SCHOOL LEVEL*'].map(lambda x: x == 'High School')
dsProgReports = dsProgReports[mask]

final = dsSATs.join(dsClassSize).join(dsProgReports).merge(dsDistrict, left_on='DISTRICT', right_index=True).merge(dsAttendEnroll, left_on='DISTRICT', right_index=True)

#print('Final = ')
#print(final[:5])
#print('*************')	


# (Even More) Additional Cleanup
# We should be in a position to build a predictive model for our target variables right away but
# unfortunately there is still messy data floating around in the dataframe that machine learning
# algorithms will choke on. A pure feature matrix should have only numeric features, but we can
# see that isn't the case. However for many of these columns, the right approach is obvious once
# we've dug in.

final.dtypes[final.dtypes.map(lambda x: x=='object')]

#print('final.dtypes = ')
#print(final.dtypes[final.dtypes.map(lambda x: x=='object')])
#print('*************')	

#Just drop string columns. [23]
#In theory we could build features out of some of these, but it is impractical here.

final = final.drop(['School Name','SCHOOL','PRINCIPAL','PROGRESS REPORT TYPE'],axis=1)
#print('final = ')
#print(final[:5])
#print('*************')	

# Remove % signs and convert to float
final['YTD % Attendance (Avg)'] = final['YTD % Attendance (Avg)'].map(lambda x: x.replace("%","")).astype(float)
#print('final no % = ')
#print(final['YTD % Attendance (Avg)'][:5])
#print('*************')	


# The last few columns we still have to deal with
final.dtypes[final.dtypes.map(lambda x: x=='object')]
#print('final.dtypes = ')
#print(final['YTD % Attendance (Avg)'][:5])
#print('*************')	

# Categorical Variables [24]
# We can see above that the remaining non-numeric field are grades . Intuitively, they might be
# important so we don't want to drop them, but in order to get a pure feature matrix we need
# numeric values. The approach we'll use here is to explode these into multiple boolean
# columns. Some machine learning libraries effectively do this for you under the covers, but
# when the cardinality of the categorical variable is relatively low, it's nice to be explicit about it.

# gradeCols = ['2009-2010 OVERALL GRADE','2009-2010 ENVIRONMENT GRADE','2009-2010 PERFORMANCE GRADE','2009-2010 PROGRESS GRADE','2008-09 PROGRESS REPORT GRADE']
gradeCols = ['2009-2010 OVERALL GRADE','2009-2010 ENVIRONMENT GRADE','2009-2010 PERFORMANCE GRADE','2009-2010 PROGRESS GRADE','2008-09 PROGRESS REPORT GRADE']

#print('gradeCols = ')
#print(gradeCols)
#print('*************')

#print('final[gradeCols] = ')
#print(final[gradeCols][:5])
#print('*************')

#[nan, A, B, C, D, F]
#grades=np.unique(final[gradeCols].values)
grades1=pd.unique(final['2009-2010 OVERALL GRADE'].values)
grades2=pd.unique(final['2009-2010 ENVIRONMENT GRADE'].values)
grades3=pd.unique(final['2009-2010 PERFORMANCE GRADE'].values)
grades4=pd.unique(final['2009-2010 PROGRESS GRADE'].values)
grades5=pd.unique(final['2008-09 PROGRESS REPORT GRADE'].values)

for g in grades1:
	final = final.join(pd.Series(data=final['2009-2010 OVERALL GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 OVERALL GRADE_is_" + str(g)))

for g in grades2:
	final = final.join(pd.Series(data=final['2009-2010 ENVIRONMENT GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 ENVIRONMENT GRADE_is_" + str(g)))

for g in grades3:
	final = final.join(pd.Series(data=final['2009-2010 PERFORMANCE GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 PERFORMANCE GRADE_is_" + str(g)))

for g in grades4:
	final = final.join(pd.Series(data=final['2009-2010 PROGRESS GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 PROGRESS GRADE_is_" + str(g)))

for g in grades5:
	final = final.join(pd.Series(data=final['2008-09 PROGRESS REPORT GRADE'].map(lambda x: 1 if x is g else 0), name = "2008-09 PROGRESS REPORT GRADE_is_" + str(g)))
	

##print('pd.Series = ')
##print(pd.Series(data=final['2009-2010 OVERALL GRADE'].map(lambda x: 1 if x is g else 0), name = "2009-2010 OVERALL GRADE_is_" + str(g)))
##print('*************')
	
#for c in gradeCols:
#	for g in grades:
#		final = final.join(pd.Series(data=final[c].map(lambda x: 1 if x is g else 0), name = c + "_is_" + str(g)))

final = final.drop(gradeCols, axis=1)

#print('final = ')
#print(final[:5])
#print('*************')

# Uncomment to generate CSV files:
final.drop(['Critical Reading Mean','Mathematics Mean','Writing Mean'],axis=1).to_csv('C:/Users/gadorey/projects/myPandas/results/train.csv')
final.filter(['Critical Reading Mean','Mathematics Mean','Writing Mean']).to_csv('C:/Users/gadorey/projects/myPandas/results/target.csv')

#That's it!
#We now have a feature matrix that's trivial to use with any number of machine learning
#algorithms. Feel free to stop here and run the two lines of code below to get nice CSV files
#written to disk that you can easily use in Excel, Tableau, etc. Or run the larger block of code to
#see how easy it is to build a random forest model against this data, and look at which variables
#are most important.

#from sklearn.ensemble import RandomForestRegressor



target=final.filter(['Critical Reading Mean'])
#print('target = ')
#print(target)
#print('*************')

# We drop all three dependent variables because we dont want them used when trying to make a prediction
train = final.drop(['Critical Reading Mean','Writing Mean','Mathematics Mean','SCHOOL LEVEL*'],axis=1)
##print('train = ')
##print(train['High School'][:5])
##print('*************')
#print('train columns = ')
#print(train.columns)
#print('*************')	

target_is_nan=np.any(np.isnan(target))
target_is_infinite=np.all(np.isfinite(target))
train_is_nan=np.any(np.isnan(train))
train_is_infinite=np.all(np.isfinite(train))
#print('target_is_nan= ',target_is_nan)
#print('target_is_infinite= ',target_is_infinite)
#print('train_is_nan= ',train_is_nan)
#print('target_is_infinite= ',train_is_infinite)
#print('*************')	
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
target2.to_csv('C:/Users/gadorey/projects/myPandas/results/target2.csv')

train2=train
where_are_NaNs = np.isnan(train2)
train2[where_are_NaNs] = 0
train2.to_csv('C:/Users/gadorey/projects/myPandas/results/train2.csv')




# model=RandomForestRegressor(n_estimators=100, n_jobs=-1, compute_importances=True)
model=RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(train2,target2)

predictions=np.array(model.predict(train2))
rmse=math.sqrt(np.mean((np.array(target2.values)-predictions)**2))
imp=sorted(zip(train2.columns,model.feature_importances_),key=lambda tup: tup[1],reverse=True)

#print('RMSE: '+str(rmse))
#print('10 Most important variables: '+str(imp[:10]))


return render(request, 'polls/predict.html', {'rmse': rmse})


####
import matplotlib.pyplot as plt

plt.plot([1,2,3,4,5],[4,5,6,7,3])

plt.show()



