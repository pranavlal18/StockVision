

# Create your models here.
# models.py
from django.db import models

class StockTicker(models.Model):
    ticker = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return self.ticker






from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(blank=True, null=True)  # Add a bio field for user profile
    location = models.CharField(max_length=100, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    
    # Add any other relevant fields for the profile
    def __str__(self):
        return self.user.username
    

from django.db import models
from django.contrib.auth.models import User

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock_ticker = models.CharField(max_length=10)
    shares = models.FloatField()
    purchase_price = models.FloatField()
    date_purchased = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.stock_ticker}"



# models.py
from django.contrib.auth.models import User
from django.db import models

class Stock(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Temporarily allow null
    ticker = models.CharField(max_length=15)

    def __str__(self):
        return self.ticker


class AnalystRating(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    recommendation = models.CharField(max_length=10)  # e.g., 'Buy', 'Hold', 'Sell'
    target_price = models.FloatField()

