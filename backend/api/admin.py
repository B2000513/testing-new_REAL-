from django.contrib import admin
from api.models import User, Profile, Customer

# Register your models here.
class UserAdmin(admin.ModelAdmin):
    list_display = ['username','email','verified']
    list_editable = ['verified']
class ProfileAdmin(admin.ModelAdmin):
    
    list_display = ['user', 'full_name',]
    

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ('customerID', 'Email', 'Churn', 'SatisfactionScore', 'MonthlyCharges', 'PaymentMethod')
    list_filter = ('Churn', 'PaymentMethod', 'Contract')
    search_fields = ('customerID', 'Email', 'PaymentMethod')
    ordering = ('-SatisfactionScore',)
    list_per_page = 50
    readonly_fields = ('customerID',)

    fieldsets = (
        ('Basic Info', {'fields': ('customerID', 'Email', 'Churn')}),
        ('Payment Details', {'fields': ('PaymentMethod', 'MonthlyCharges', 'Contract')}),
        ('Satisfaction Metrics', {'fields': ('SatisfactionScore', 'LifetimeValue')}),
    )




admin.site.register(User, UserAdmin)
admin.site.register(Profile, ProfileAdmin)