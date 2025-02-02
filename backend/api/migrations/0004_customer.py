# Generated by Django 5.1.3 on 2025-02-02 09:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_remove_profile_verified_user_verified'),
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('customerID', models.CharField(max_length=255)),
                ('gender', models.IntegerField()),
                ('SeniorCitizen', models.IntegerField()),
                ('Partner', models.IntegerField()),
                ('Dependents', models.IntegerField()),
                ('tenure', models.IntegerField()),
                ('PhoneService', models.IntegerField()),
                ('MultipleLines', models.IntegerField()),
                ('InternetService', models.CharField(max_length=255)),
                ('OnlineSecurity', models.IntegerField()),
                ('OnlineBackup', models.IntegerField()),
                ('DeviceProtection', models.IntegerField()),
                ('TechSupport', models.IntegerField()),
                ('StreamingTV', models.IntegerField()),
                ('StreamingMovies', models.IntegerField()),
                ('Contract', models.CharField(max_length=255)),
                ('PaperlessBilling', models.IntegerField()),
                ('PaymentMethod', models.CharField(max_length=255)),
                ('MonthlyCharges', models.FloatField()),
                ('TotalCharges', models.FloatField()),
                ('Churn', models.IntegerField()),
                ('Age', models.IntegerField()),
                ('SatisfactionScore', models.IntegerField()),
                ('CustomerSupportCalls', models.IntegerField()),
                ('PaymentTimeliness', models.IntegerField()),
                ('LifetimeValue', models.FloatField()),
                ('AverageDailyUsage', models.FloatField()),
                ('Email', models.EmailField(max_length=254)),
            ],
        ),
    ]
