from django.contrib import admin

# Register your models here.

from .models import Profile

# Register the Profile model so that it shows up in the admin panel
admin.site.register(Profile)


