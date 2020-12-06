from django.contrib import admin
from tensor.models import Tensor, TestResult

# Register your models here.
admin.site.register(Tensor)
admin.site.register(TestResult)