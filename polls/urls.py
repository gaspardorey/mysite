from django.conf.urls import url, patterns, include

from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = 'polls'
urlpatterns = [
	url(r'^$', views.IndexView.as_view(), name='index'),
	url(r'^(?P<pk>[0-9]+)/$', views.DetailView.as_view(), name='detail'),
	url(r'^(?P<pk>[0-9]+)/results/$', views.ResultsView.as_view(), name='results'),
	url(r'^(?P<question_id>[0-9]+)/vote/$', views.vote, name='vote'),
	url(r'^predict/$', views.predict, name='predict'),
	url(r'^promotions/$', views.promotions, name='promotions'),
	url(r'^holt/$', views.inputs, name='holt'),
	url(r'^linechart/$', views.demo_linechart, name='demo_linechart'),

	#url(r'^predict/$', views.PredictView.as_view(), name='predict'),
	url(r'^list/$', views.list, name='list'),
	url(r'^accounts/', include('registration.backends.default.urls')),
]

#urlpatterns = patterns('',
#    (r'^', include('polls.urls')),
#		) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# The long code...
	
	# ex: /polls/
#	url(r'^$', views.index, name='index'),
	#ex: /polls/5/
#	url(r'^(?P<question_id>[0-9]+)/$', views.detail, name='detail'),
	# ex: /polls/5/results/
#	url(r'^(?P<question_id>[0-9]+)/results/$', views.results, name='results'),
	# ex: /polls/5/vote/
#	url(r'^(?P<question_id>[0-9]+)/vote/$', views.vote, name='vote'),
	# added the word 'specifics'
	#url(r'^specifics/(?P<question_id>[0-9]+)/$', views.detail, name='detail'),
#	]