from django.db import models

# Create your models here.

# 전체 테스트 로직 반복 카운트
numberOfTest = 0

# 한번 테스트할때마다 정의할 객체 정보
class Tensor(models.Model):
    def __init__(self):
        global numberOfTest
        self.images = models.ArrayField(models.ImageField())
        self.testResult = models.DecimalField(default=0.0)
        self.isTested = models.BooleanField(default=False)
        numberOfTest = numberOfTest + 1
        
    def __str__(self):
        return self.testResult
    
class TestResult(models.Model):
    def __init__(self):
        self.results = models.ArrayField(models.DecimalField())
        
    def __str__(self):
        return self.results
    
        