from django import forms
from pages.models import Document
from pages.models import Segment

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ( 'document', )


class SegmentForm(forms.ModelForm):
    class Meta:
        model = Segment
        fields = ( 'start','end', )
