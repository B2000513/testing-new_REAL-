import pandas as pd
import joblib
from django.core.files.storage import FileSystemStorage
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
from .models import Customer

# Load pre-trained churn prediction model and feature columns
MODEL_PATH = 'churn_model.pkl'
FEATURE_COLUMNS_PATH = 'X_train_columns.pkl'
model = joblib.load(MODEL_PATH)
X_train_columns = joblib.load(FEATURE_COLUMNS_PATH)

# Preprocessing function
def preprocess_data(df):
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 
                      'PaperlessBilling', 'PaymentTimeliness', 'gender', 'SatisfactionScore']
    
    df[binary_columns] = df[binary_columns].replace({
        'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 
        'On-time': 1, 'Late': 0, 'Male': 1, 'Female': 0, 'High': 1, 'Low': 0
    })
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Handle missing columns
    missing_cols = set(X_train_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    
    # Ensure correct column order
    df_encoded = df_encoded[X_train_columns]
    
    return df_encoded

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_customers_from_excel(request):
    if 'excel_file' not in request.FILES:
        return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)
    
    excel_file = request.FILES['excel_file']
    fs = FileSystemStorage()
    filename = fs.save(excel_file.name, excel_file)
    uploaded_file_path = fs.path(filename)
    
    try:
        df = pd.read_excel(uploaded_file_path)
        df = preprocess_data(df)  # Preprocess data for prediction
        df['Churn_Prediction'] = model.predict(df)
        df['Churn_Probability'] = model.predict_proba(df)[:, 1]
        
        with transaction.atomic():  # Ensure atomicity
            Customer.objects.all().delete()  # Clear existing records
            unique_emails = set()
            customers = []
            
            for _, row in df.iterrows():
                email = str(row.get('Email', '')).strip()
                if not email or email in unique_emails:
                    continue  
                unique_emails.add(email)
                
                customers.append(Customer(
                    customerID=row.get('customerID', ''),
                    gender=row.get('gender', 0),
                    SeniorCitizen=row.get('SeniorCitizen', 0),
                    Partner=row.get('Partner', 0),
                    Dependents=row.get('Dependents', 0),
                    tenure=row.get('tenure', 0),
                    PhoneService=row.get('PhoneService', 0),
                    MultipleLines=row.get('MultipleLines', 0),
                    InternetService=row.get('InternetService', ''),
                    OnlineSecurity=row.get('OnlineSecurity', 0),
                    OnlineBackup=row.get('OnlineBackup', 0),
                    DeviceProtection=row.get('DeviceProtection', 0),
                    TechSupport=row.get('TechSupport', 0),
                    StreamingTV=row.get('StreamingTV', 0),
                    StreamingMovies=row.get('StreamingMovies', 0),
                    Contract=row.get('Contract', ''),
                    PaperlessBilling=row.get('PaperlessBilling', 0),
                    PaymentMethod=row.get('PaymentMethod', ''),
                    MonthlyCharges=row.get('MonthlyCharges', 0.0),
                    TotalCharges=row.get('TotalCharges', 0.0),
                    Churn=row.get('Churn_Prediction', 0),
                    ChurnProbability=row.get('Churn_Probability', 0.0),
                    Age=row.get('Age', 0),
                    SatisfactionScore=row.get('SatisfactionScore', 0),
                    CustomerSupportCalls=row.get('CustomerSupportCalls', 0),
                    PaymentTimeliness=row.get('PaymentTimeliness', 0),
                    LifetimeValue=row.get('LifetimeValue', 0.0),
                    AverageDailyUsage=row.get('AverageDailyUsage', 0.0),
                    Email=email
                ))
            
            Customer.objects.bulk_create(customers)  # Bulk insert for efficiency
        
        return Response({"message": "Data imported and predictions saved successfully!"}, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        return Response({"error": f"Error occurred: {e}"}, status=status.HTTP_400_BAD_REQUEST)
    
    finally:
        if fs.exists(filename):
            fs.delete(filename)
