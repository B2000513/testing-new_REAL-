from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
# Create your models here.

class User(AbstractUser):
    username = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    verified = models.BooleanField(default=False)

    def __str__(self):
        return self.username
    
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    bio =models.CharField(max_length=300)
    image = models.ImageField(default="default.jpg",upload_to='user_images')
    

    def __str__(self):
        return self.full_name
    
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)



def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()

post_save.connect(create_user_profile, sender=User)
post_save.connect(save_user_profile, sender=User)


class Customer(models.Model):
    customerID = models.CharField(max_length=255)
    gender = models.IntegerField()
    SeniorCitizen = models.IntegerField()
    Partner = models.IntegerField()
    Dependents = models.IntegerField()
    tenure = models.IntegerField()
    PhoneService = models.IntegerField()
    MultipleLines = models.IntegerField()
    InternetService = models.CharField(max_length=255)
    OnlineSecurity = models.IntegerField()
    OnlineBackup = models.IntegerField()
    DeviceProtection = models.IntegerField()
    TechSupport = models.IntegerField()
    StreamingTV = models.IntegerField()
    StreamingMovies = models.IntegerField()
    Contract = models.CharField(max_length=255)
    PaperlessBilling = models.IntegerField()
    PaymentMethod = models.CharField(max_length=255)
    MonthlyCharges = models.FloatField()
    TotalCharges = models.FloatField()
    Churn = models.IntegerField()
    Age = models.IntegerField()
    SatisfactionScore = models.IntegerField()
    CustomerSupportCalls = models.IntegerField()
    PaymentTimeliness = models.IntegerField()
    LifetimeValue = models.FloatField()
    AverageDailyUsage = models.FloatField()
    Email = models.EmailField()

    def __str__(self):
        return self.customerID

