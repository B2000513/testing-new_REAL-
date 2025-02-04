from django.contrib import admin
from api.models import User, Profile, Customer

# Register your models here.
class UserAdmin(admin.ModelAdmin):
    list_display = ['username','email','verified']
    list_editable = ['verified']
class ProfileAdmin(admin.ModelAdmin):
    
    list_display = ['user', 'full_name',]
    

@admin.register(Customer)
class Customer(admin.ModelAdmin):
    list_display = ('customerID', 'Email', 'Churn', 'SatisfactionScore', 'MonthlyCharges', 'PaymentMethod')
    list_filter = ('Churn', 'PaymentMethod', 'Contract')
    search_fields = ('customerID', 'Email', 'PaymentMethod')
    ordering = ('-SatisfactionScore',)




admin.site.register(User, UserAdmin)
admin.site.register(Profile, ProfileAdmin)