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

    #### Testing for list to update if deleted:
	
    docId = request.POST.get('doc-id', None)
    if docId is not None:
        docToDel = Document.objects.get(pk=docId)

        # delete the file using docToDel.docfile
		docToDel.delete()

		return HttpResponse('Whatever message you want.')
	
	##############	
	
	
	
	# Load documents for the list page
    documents = Document.objects.all()
	
    # Render list page with the documents and the form
    return render(request, 'polls/list.html',
			{'documents': documents, 'form': form},
			context_instance=RequestContext(request)
			)
