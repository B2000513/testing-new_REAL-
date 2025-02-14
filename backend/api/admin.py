from django.contrib import admin
from django.db import transaction
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

    actions = ['clear_customer_data']

    def clear_customer_data(self, request, queryset):
        try:
            with transaction.atomic():
                Customer.objects.all().delete()
            self.message_user(request, "All customer data has been cleared.", messages.SUCCESS)
        except Exception as e:
            self.message_user(request, f"Error clearing data: {e}", messages.ERROR)

    clear_customer_data.short_description = "Clear all customer data"




admin.site.register(User, UserAdmin)
admin.site.register(Profile, ProfileAdmin)