from django.db import models

# Create your models here.
class Document(models.Model):
    #description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='files/')


class Segment(models.Model):
    start = models.IntegerField()
    end = models.IntegerField()
