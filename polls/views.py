from __future__ import division # for the def holt(request)

from django.shortcuts import render, get_object_or_404, render_to_response
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.template import RequestContext
from django.core.urlresolvers import reverse
from django.views import generic
from django.utils import timezone
from pandas import * # for Triple Exponential
import pandas as pd
import numpy as np
import math
import re
from sklearn.ensemble import RandomForestRegressor

from .models import Document
from .forms import DocumentForm

from .models import Choice, Question

from sys import exit # for the def holt(request)
from math import sqrt # for the def holt(request)
from numpy import array # for the def holt(request)
from scipy.optimize import fmin_l_bfgs_b # for the def holt(request)
import matplotlib.pyplot as plt # for the def holt(request)

import random # for def demo_linechart
import datetime # for def demo_linechart
import time # for def demo_linechart

import plotly.tools as tls # for def demo_linechart v2
tls.set_credentials_file(username='gaspardorey',api_key='fobabamwnw') # for def demo_linechart v2

from sqlalchemy import create_engine # database connection  # # for def demo_linechart
import datetime as dt # for def demo_linechart
import sqlite3 # to delete database # # for def demo_linechart

import plotly.plotly as py # interactive graphing # # for def demo_linechart
from plotly.graph_objs import Bar, Scatter, Marker, Layout # for def demo_linechart
import plotly.graph_objs as go # for def demo_linechart

class IndexView(generic.ListView):
	template_name = 'polls/base.html' #index.html'
	context_object_name = 'latest_question_list'
	
	def get_queryset(self):
		"""Return the last five published questions (not including those set to be
		published in the future
		"""
		return Question.objects.filter(
			pub_date__lte=timezone.now()
		).order_by('-pub_date')[:5]
		#return Question.objects.order_by('-pub_date')[:5]
		
		
class DetailView(generic.DetailView):
	model = Question
	template_name = 'polls/detail.html'

	def get_queryset(self):
		"""
		Excludes any questions that aren't published yet.
		"""
		return Question.objects.filter(pub_date__lte=timezone.now())

	
class ResultsView(generic.DetailView):
	model = Question
	template_name = 'polls/results.html'
	

def vote(request, question_id):
#	return HttpResponse("You're voting on question %s." % question_id)
		question = get_object_or_404(Question, pk=question_id)
		try:
			selected_choice = question.choice_set.get(pk=request.POST['choice'])
		except (KeyError, Choice.DoesNotExist):
			# Redisplay the question voting form.
			return render(request, 'polls/detail.html', {
				'question': question,
				'error_message': "You didn't select a choice.",
			})
		else:
			selected_choice.votes += 1
			selected_choice.save()
			# Always return an HttpResponseRedirect after successfully dealing
			# with POST data. This prevents data from being posted twice if a 
			# user hits the Back button.
			return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))




# Create your views here.

#def index(request):
#	return HttpResponse("Hello, world. You're at the polls index.")

#def index(request):
#	latest_question_list = Question.objects.order_by('-pub_date')[:5]
#	# We could leave this in but we would be hard coding items
#	# It is better to take the design out:
#	# output = ', '.join([q.question_text for q in latest_question_list])
#	template = loader.get_template('polls/index.html')
#	context = {
#		'latest_question_list': latest_question_list,
#		}
#	return HttpResponse(template.render(context, request))

#def index(request):
#	latest_question_list = Question.objects.order_by('-pub_date')[:5]
#	context = {'latest_question_list': latest_question_list}
#	return render(request, 'polls/index.html', context)
	
	
#def detail(request, question_id):
#	return HttpResponse("You're looking at question %s." % question_id)
#	try:
#		question = Question.objects.get(pk=question_id)
#	except Question.DoesNotExist:
#		raise Http404("Question does not exist")
#	question = get_object_or_404(Question, pk=question_id)
#	return render(request, 'polls/detail.html', {'question': question})
	
#def results(request, question_id):
#	question = get_object_or_404(Question, pk=question_id)
#	return render(request, 'polls/results.html', {'question':question})

   
		
def predict(request):
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
	nrmse=round(100*rmse/np.mean(target2.values),2)
	#return HttpResponse(rmse, request)
	#return HttpResponse(request, rmse)
	return render(request, 'polls/predict.html', {'rmse': round(rmse,2),'nrmse':nrmse})
	
	#return render(request, 'polls/detail.html', rmse)
	#	else:
	#		selected_choice.votes += 1
	#		selected_choice.save()
	#		# Always return an HttpResponseRedirect after successfully dealing
	#		# with POST data. This prevents data from being posted twice if a 
	#		# user hits the Back button.
	#		return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))



def promotions(request):
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

	return render(request, 'polls/promotions.html', {'rmse': round(rmse,2), 'imp': imp})


	

# Uploading a file:
def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('polls.views.list'))
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()
	
    # Render list page with the documents and the form
    return render(request, 'polls/list.html',
			{'documents': documents, 'form': form},
			context_instance=RequestContext(request)
			)

			
	
########################################

# Holt-Winters:
'''	
def RMSE(params, *args):
    """
    Root-Mean-Square Error   
    """
    series, d = args
    a, b, g = params
    X = holtWinters(series, d, 0, a, b, g)
    return sqrt(sum([(a - b) ** 2 for a, b in zip(series[1:], X)]) / len(series))

def holtWinters(series, d, f, a = None, b = None, g = None):
    """
    The Holt-Winters Forecasting Method
    parameters:
        series -- series to forecast
        d -- seasonal period 
        f -- forecast period 
        a -- level component 
        b -- trend component 
        g -- seasonal component    
    returns: 
        a list of forecast values
    """
    x = series[:]
    L, T, I, X = ([], [], [], [])

    # use 1st period of data for initialization
    r = np.polyfit(range(d), x[:d], 1) if d > 1 else [x[0], 0] # linear regression for 1st period of data
    for t in range(d):
        L.append(r[0] * (t + 1) + r[1])
        T.append(0.0)
        I.append(x[t] - L[t])
        X.append(L[t] + T[t] + I[t])

    # optimize by root-mean-square error using L-BFGS-B algorithm
    if (a == None or b == None or g == None): 
        a, b, g = fmin_l_bfgs_b(RMSE, 
            x0 = (0.3, 0.1, 0.1), 
            args = (x, d), 
            bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 
            approx_grad = True)[0]
    
    # one step forward forecast
    for t in range(d, len(x) - 1 + f):
        if t == len(x):
            # if there is no more data in time series then append forecast value
            x.append(X[-1])
        L.append(a * (x[t] - I[t - d]) + (1 - a) * (L[t - 1] + T[t - 1]))
        T.append(b * (L[t] - L[t - 1]) + (1 - b) * T[t - 1])
        I.append(g * (x[t] - L[t]) + (1 - g) * I[t - d])
        X.append(L[t] + T[t] + I[t - d + 1]) # L[t] + h * T[t] + I[t - d + h % d]        
        
    return X
   
    
def main(request):
    
	
	series = [5771, 5431, 5741, 4505, 4485, 7682, 6946, 6772, 7103, 6878, 6382, 5198, 9018, 8971, 8653, 5170, 5725, 4240, 4144, 9217, 7151, 6699, 6633, 6677, 4590, 3752, 6742, 5740, 5616, 5358, 4787, 3035, 2037, 5263, 7008, 6188, 6124, 6087, 4863, 3557, 6808, 6415, 6355, 6175, 6235, 4232, 3652, 6737, 6474, 5881, 5845, 5379, 4538, 3701, 6270, 5266, 5246, 5325, 5884, 4430, 3929, 6421, 6128, 5864, 5579, 5964, 5637, 4470, 7324, 6972, 6702, 6506, 6957, 6954, 5451, 7993, 7204, 6512, 6835, 6573, 5928, 5000, 7406, 6899, 6591, 6208, 6330, 5444, 4351, 7592, 6775, 6361, 6937, 6290, 6134, 4545, 5242, 5072, 5502, 4938, 5475, 5045, 3467, 6147, 5761, 5260]
	
	X = holtWinters(series, 7, 30)
	
	
	# Plot with matplotlib:
	fig1, ax1 = plt.subplots(figsize = (10, 6))
	ax1.plot(range(len(series)), series, 'rs-', label = 'actual')
	ax1.plot(range(1, len(X) + 1), X, 'co--', label = 'forecast')
	ax1.legend(loc = 'best')
	ax1.grid()
	plt.show()
	
	
	return render(request, 'polls/holt.html', {'rmse': X})

if __name__ == "__main__":
    main()
'''
###############
	
def demo_linechart(request):
    """
    lineChart page
    """
    start_time = int(time.mktime(datetime.datetime(2012, 6, 1).timetuple()) * 1000)
    nb_element = 150
    xdata = range(nb_element)
    xdata = map(lambda x: start_time + x * 1000000000, xdata)
    ydata = [i + random.randint(1, 10) for i in range(nb_element)]
    ydata2 = map(lambda x: x * 2, ydata)

    tooltip_date = "%d %b %Y %H:%M:%S %p"
    extra_serie1 = {
        "tooltip": {"y_start": "", "y_end": " cal"},
        "date_format": tooltip_date,
        'color': '#a4c639'
    }
    extra_serie2 = {
        "tooltip": {"y_start": "", "y_end": " cal"},
        "date_format": tooltip_date,
        'color': '#FF8aF8'
    }
    chartdata = {'x': xdata,
                 'name1': 'series 1', 'y1': ydata, 'extra1': extra_serie1,
                 'name2': 'series 2', 'y2': ydata2, 'extra2': extra_serie2}

    charttype = "lineChart"
    chartcontainer = 'linechart_container'  # container name
    data = {
        'charttype': charttype,
        'chartdata': chartdata,
        'chartcontainer': chartcontainer,
        'extra': {
            'x_is_date': True,
            'x_axis_format': '%d %b %Y %H',
            'tag_script_js': True,
            'jquery_on_ready': False,
        }
    }
    return render_to_response('polls/linechart.html', {'linechart':data})
    #return render_to_response('polls/linechart.html', data)
	#return render(request, 'polls/holt.html', {'rmse': X})
	

#######################################

def TripleExponentialSmoothing(Input,N,Alpha,Beta,Gamma,Seasons,test=0):

    # NOTE: Any multiplication done by 1.0 is to prevent numbers from getting rounded to int
    Num = N
    a = Alpha
    b = Beta
    c = Gamma
    s = Seasons

    # Create data frame (previously changed Sales and Period)
    df = DataFrame(Input, columns=['Period'])
    df['Sales'] = DataFrame(Input, columns=['Sales'])
	
    # Add Cummulative sums
    df['CummSum'] = df['Sales'].cumsum()

    # Initial values for L
    '''  L = average of the first period (s)
           = (x1+x2+x3+...+xs) / s

             **start calculating L at time s
    '''
    df['L'] = df['Sales'][:s].mean() # Level initial
    df['L'][:s-1] = None #erase all values before period s

    # Initial values for b
    '''     b = (current value - first value) / ( period(s) - 1 )

             **start calculating b at time s
    '''
	#df['b'] = 1.0*(df['Sales'][s-1] - df['Sales'][0]) / (s - 1)
	
    df['b'] = 1.0 * 2
    df['b'][:s-1] = None #erase all values before period s
	
	# Add initial seasonality for period s
    '''     s = (current value) / (average of the first period(s) )

             **only calculate for the first period s
    '''
    df['s'] = 1.0*df['Sales'][:s-1]/df['Sales'][:s].mean()

    # Initial value at time s-1
    # this is exactly the row after the previous "df['s'] =" statement 
    df['s'][s-1] = 1.0*df['Sales'][s-1]/df['Sales'][:s].mean()
    
    # Initial forecast = actual
    '''     It does not matter what number you set the initial forecast,
            we only do this to create the column
    '''
    df['TripleExpo'] = df['Sales'][0]*1.0
    df['TripleExpo'][:s] = None #erase all values before and including period s


    # Triple Exponential Smoothing
    for i in range(Num):

        # We start at the end of period s
        if (i >= s):

            # Get the previous L and b
            LPrev = df['L'][i-1]
            bPrev = df['b'][i-1]
            #print LPrev, bPrev

            '''
                Eq1. L1 = alpha * (y1 /	S0) + (1 - alpha) * (L0 + b0)
                Eq2. b1 = beta * (L1 - L0) + (1 - beta) * b0
                Eq3. S1 = Gamma * (y1 / L1) + (1 - Gamma) * S0
                Eq4. F(1+m) = (L1 + b1) * S0)
            '''
            df['L'][i] = a * (df['Sales'][i] / df['s'][i-s]) + ((1 - a) * (LPrev + bPrev))
            df['b'][i] = b * (df['L'][i] - LPrev) + (1 - b) * bPrev
            df['s'][i] = c * (df['Sales'][i] / df['L'][i]) + (1 - c) * df['s'][i-s]
            #print ( df['L'][i], df['b'][i], df['s'][i])

            # forecast for period i
            df['TripleExpo'][i] = (df['L'][i-1] + df['b'][i-1]) * df['s'][i-s]
            #print (df['TripleExpo'][i])


    # Track Errors
    df['TresError'] = df['Sales'] - df['TripleExpo']
    df['TresMFE'] = (df['TresError']).mean()
    df['TresMAD'] = np.fabs(df['TresError']).mean()
    df['TresMSE'] = np.sqrt(np.square(df['TresError']).mean())
    df['TresTS'] = np.sum(df['TresError'])/df['TresMAD']

    #print (a,b, df.TresMAD[0])

    if test == 0:
        return df.TresMAD[0]
    else: return df



#----------------------------------------------------
#-------- INPUT DATA ------------
#----------------------------------------------------

# Input Data:
def inputs(request):
	original=pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/holt_sales.csv')
	disk_engine = create_engine('sqlite:///holt_sales.db')
	
	####
	#### Delete all previous records
	conn = sqlite3.connect('holt_sales.db')
	#print("Connection OK")

	df = pd.read_sql_query('SELECT * FROM data_sql', disk_engine)
	#print("df = ")
	#print(df[:5])

	c = conn.cursor()
	mydata = c.execute("DELETE FROM data_sql")
	conn.commit()
	c.close
	#print("All previous records deleted.")
	#####
	
	start = dt.datetime.now()
	chunksize = 20 #20000 for 311_100M
	j = 0
	index_start = 1

	for df in pd.read_csv('C:/Users/gadorey/projects/mysite/polls/media/documents/holt_sales.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):
		df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns
		df['Period'] = pd.to_datetime(df['Period']) # Convert to datetimes
		df.index += index_start
		j+=1
		df.to_sql('data_sql', disk_engine, if_exists='append')
		index_start = df.index[-1] + 1
		
	df = pd.read_sql_query('SELECT * FROM data_sql', disk_engine)
	
	data = df.Sales
	
	X = TripleExponentialSmoothing(data,50,0.3,0.3,0.3,3,2)
	#print('Triple Exponential = ')
	#print(Try)
	
	
	#Chart:
	# Create traces
	trace0 = go.Scatter(
		x = df.Period,
		y = X.Sales,
		mode = 'lines',
		name = 'Sales'
	)
	trace1 = go.Scatter(
		x = df.Period,
		y = X.TripleExpo,
		mode = 'lines',
		name = 'Forecast'
	)
	
	data_chart = [trace0, trace1]
	
	
	#py.iplot([Scatter(x=df.Period, y=df.Sales)], mode='lines+markers', filename='holt_sales/most common sales by sales', name="'Sales'", hoverinfo='name',line=dict(shape='linear'))
	py.iplot(data_chart, filename='holt_sales/most common sales by sales')

	return render(request, 'polls/holt.html', {'rmse': X})

