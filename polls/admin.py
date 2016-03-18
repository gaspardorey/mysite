

from django.contrib import admin

# Register your models here.

from .models import Choice, Question

# admin.site.register(Question)

## You’ll follow this pattern – create a model admin class, then pass it 
# as the second argument to admin.site.register() – any time you need to 
# change the admin options for an model. This particular change above makes 
# the “Publication date” come before the “Question” field:

# class ChoiceInline(admin.StackedInline): # this option takes a lot of space
class ChoiceInline(admin.TabularInline):
	model = Choice
	extra = 3
	
class QuestionAdmin(admin.ModelAdmin):
#	fields = ['pub_date', 'question_text']
	fieldsets = [
		(None, 					{'fields': ['question_text']}),
		('Date information', 	{'fields': ['pub_date']}),
	]
	inlines = [ChoiceInline]
	list_display = ('question_text', 'pub_date', 'was_published_recently')
	list_filter = ['pub_date']
	search_fields = ['question_text']
	
admin.site.register(Question, QuestionAdmin)

# admin.site.register(Choice)

