import pandas as pd
import openai # Install with `pip install openai`
import os
import pickle
from dotenv import load_dotenv
from django.shortcuts import render , redirect
import joblib


# Create your views here.
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from api.models import User, Profile ,Customer
from api.serializers import UserSerializer, MyTokenObtainPairSerializer, RegisterSerializer ,ProfileSerializer ,CustomerSerializer
from rest_framework.decorators import api_view,permission_classes
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics,status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings
from .serializers import ChangePasswordSerializer
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db import transaction
from django.http import JsonResponse
from .models import Customer  # Adjust based on your chatbot's needs
from .serializers import CustomerSerializer  # Use a lightweight serializer
from .llama_model import generate_response  # ✅ Import LLaMA chatbot function
    

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer


@api_view(["GET"])
def getRoutes(request):
     routes =[
          '/api/token/',
        '/api/register/',
        '/api/token/refresh/',
        '/api/profile/',
        '/api/profile/update/',
        '/api/upload/',
        '/api/customers/',
     ]
     return Response(routes)


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def testEndPoint(request):
    """Test endpoint for authenticated users to check access."""
    # Get the user's full name if available, otherwise use the username
    full_name = getattr(request.user.profile, 'full_name', request.user.username)

    if request.method == "GET":
        response = f"Welcome {full_name}"  # Use full name or username here
        return Response({'response': response}, status=status.HTTP_200_OK)

    elif request.method == "POST":
        text = request.data.get('text', '')  # Use request.data for better DRF handling
        response = f"Welcome {full_name}, you said {text}"  # Include full name here as well
        return Response({'response': response}, status=status.HTTP_200_OK)

    return Response({'response': 'Invalid request'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_profile(request):
    """Retrieve the profile of the authenticated user."""
    try:
        profile = Profile.objects.get(user=request.user)
        serializer = ProfileSerializer(profile)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Profile.DoesNotExist:
        return Response({"detail": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """Update the profile of the authenticated user."""
    try:
        profile = Profile.objects.get(user=request.user)
        # Handle image upload if present
        if 'image' in request.FILES:
            image = request.FILES['image']
            image_path = default_storage.save(f"profile_images/{image.name}", image)
            image_url = f"{settings.MEDIA_URL}{image_path}"  # Construct image URL
        
            # Include the image in the request data for the serializer
            request.data._mutable = True  # Allows modification of request.data if it's immutable
            request.data['image'] = image_url
            request.data._mutable = False  # Set it back to immutable


        serializer = ProfileSerializer(profile, data=request.data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Profile.DoesNotExist:
        return Response({"detail": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    serializer = ChangePasswordSerializer(data=request.data, context={'request': request})
    if serializer.is_valid():
        request.user.set_password(serializer.validated_data['new_password'])
        request.user.save()
        return Response({"detail": "Password changed successfully."}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PasswordResetRequestView(APIView):
    def post(self, request, *args, **kwargs):
        email = request.data.get("email")
        if not email:
            return Response({"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.filter(email=email).first()
        if user:
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_link = f"{request.scheme}://{request.get_host()}/reset-password/{uid}/{token}/"
            
            context = {
                "reset_link": reset_link,
                "user": user,
            }
            subject = "Password Reset Request"
            message = render_to_string("registration/password_reset_email.html", context)
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email])

        # Always return success to prevent email enumeration
        return Response({"message": "If an account with this email exists, a reset link has been sent."}, status=status.HTTP_200_OK)
    

# Load the trained model and feature columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'churn_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'X_train_columns.pkl')

# Load model and features
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

@csrf_exempt
def upload_customers_from_csv(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        file = request.FILES['file']
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'Invalid file format, please upload a CSV file'}, status=400)

        # Save temporary file
        file_path = default_storage.save('temp/' + file.name, ContentFile(file.read()))
        df = pd.read_csv(file_path)
        os.remove(file_path)  # Clean up the temporary file

        # Ensure all required columns are present
        required_columns = [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Age",
            "SatisfactionScore", "CustomerSupportCalls", "PaymentTimeliness",
            "LifetimeValue", "AverageDailyUsage", "Email"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return JsonResponse({'error': f'Missing columns: {missing_columns}'}, status=400)

        if df.empty:
            return JsonResponse({'error': 'CSV file is empty or corrupt'}, status=400)

        # Convert categorical binary values
        binary_columns = [
            "Partner", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "PaperlessBilling",
            "PaymentTimeliness", "gender", "SatisfactionScore"
        ]

        df[binary_columns] = df[binary_columns].replace({
            'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0,
            'On-time': 1, 'Late': 0, 'Male': 1, 'Female': 0, 'High': 1, 'Low': 0
        })

        # One-hot encode categorical columns
        categorical_cols = ["InternetService", "Contract", "PaymentMethod"]
        df = pd.get_dummies(df, columns=categorical_cols)

        # Add missing columns and ensure order
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing feature columns with default value

        df = df[feature_columns]  # Ensure correct column order

        # Predict churn probability and classify churn (1 for churned, 0 for not churned)
        churn_probs = model.predict_proba(df)[:, 1]
        churn_predictions = (churn_probs >= 0.5).astype(int)  # Threshold at 0.5

        # Store data in database
        customers = []
        for i, row in df.iterrows():
            customer = Customer(
                customerID=row["customerID"],
                gender=row["gender"],
                SeniorCitizen=row["SeniorCitizen"],
                Partner=row["Partner"],
                Dependents=row["Dependents"],
                tenure=row["tenure"],
                PhoneService=row["PhoneService"],
                MultipleLines=row["MultipleLines"],
                InternetService="Unknown",  # Categorical data converted
                OnlineSecurity=row["OnlineSecurity"],
                OnlineBackup=row["OnlineBackup"],
                DeviceProtection=row["DeviceProtection"],
                TechSupport=row["TechSupport"],
                StreamingTV=row["StreamingTV"],
                StreamingMovies=row["StreamingMovies"],
                Contract="Unknown",  # Categorical data converted
                PaperlessBilling=row["PaperlessBilling"],
                PaymentMethod="Unknown",  # Categorical data converted
                MonthlyCharges=row["MonthlyCharges"],
                TotalCharges=row["TotalCharges"],
                Churn=churn_predictions[i],  # Assign predicted churn value
                Age=row["Age"],
                SatisfactionScore=row["SatisfactionScore"],
                CustomerSupportCalls=row["CustomerSupportCalls"],
                PaymentTimeliness=row["PaymentTimeliness"],
                LifetimeValue=row["LifetimeValue"],
                AverageDailyUsage=row["AverageDailyUsage"],
                Email=row["Email"],
                churn_probability=churn_probs[i]  # Store probability
            )
            customers.append(customer)

        Customer.objects.bulk_create(customers)

        return JsonResponse({'message': f'{len(customers)} customers added with churn predictions'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])  # Protect API access
@permission_classes([IsAuthenticated])
def get_customers(request):
    try:
        customers = Customer.objects.all()  # Fetch all customers
        serializer = CustomerSerializer(customers, many=True)  # Serialize the data
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
@permission_classes([IsAuthenticated])
class CustomerListView(APIView):
    def get(self, request):
        # Debug: Log user making the request
        print(f"User: {request.user}, Authenticated: {request.user.is_authenticated}")

        customers = Customer.objects.all()
        data = [{"id": c.id, "email": c.email, "churn": c.churn} for c in customers]
        return Response(data)
    

@api_view(['GET'])
def chatbot(request):
    user_message = request.GET.get("message", "")

    if not user_message:
        return JsonResponse({"response": "Please provide a message."})

    try:
        bot_reply = generate_response(user_message)  # ✅ Call LLaMA model
        return JsonResponse({"response": bot_reply})

    except Exception as e:
        return JsonResponse({"error": f"LLaMA Error: {str(e)}"}, status=500)


@api_view(['GET'])
def chatbot_customers(request):
    # Only retrieve minimal data needed for chatbot (e.g., names, last interactions)
    customers = Customer.objects.values('id', 'name', 'last_interaction')[:50]  # Limit results
    return Response({'customers': list(customers)}) 


X_train_columns = [
    'gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService_Fiber optic', 'InternetService_No',  # Fixed this
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'MonthlyCharges', 'TotalCharges', 'Age', 'SatisfactionScore',
    'CustomerSupportCalls', 'PaymentTimeliness', 'LifetimeValue', 'AverageDailyUsage'
]

MODEL_PATH = "api/churn_model.pkl"
FEATURES_PATH = "api/X_train_columns.pkl"

model = joblib.load(MODEL_PATH)
X_train_columns = joblib.load(FEATURES_PATH)

model = joblib.load(MODEL_PATH)
X_train_columns = joblib.load(FEATURES_PATH)

@csrf_exempt  # Remove in production, use authentication
def upload_and_predict(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Read CSV file
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                return JsonResponse({'error': 'Uploaded CSV file is empty'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Invalid CSV file: {str(e)}'}, status=400)

        # 🔹 Step 1: Convert binary categorical columns
        binary_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'PaperlessBilling', 'PaymentTimeliness', 'gender', 'SatisfactionScore']
        
        df[binary_columns] = df[binary_columns].replace({
            'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 
            'On-time': 1, 'Late': 0, 'Male': 1, 'Female': 0, 'High': 1, 'Low': 0
        })

        # 🔹 Step 2: One-hot encoding
        df_encoded = pd.get_dummies(df, drop_first=True)

        # 🔹 Step 3: Ensure feature names match training set
        missing_cols = set(X_train_columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0  # Add missing columns

        extra_cols = set(df_encoded.columns) - set(X_train_columns)
        if extra_cols:
            print("⚠️ Extra columns removed:", extra_cols)
            df_encoded.drop(columns=extra_cols, inplace=True)

        # 🔹 Step 4: Ensure correct column order
        df_encoded = df_encoded[X_train_columns].astype(float)

        # 🔹 Step 5: Predict churn
        df['Churn_Prediction'] = model.predict(df_encoded)
        df['Churn_Probability'] = model.predict_proba(df_encoded)[:, 1]

        # 🔹 Step 6: Handle duplicate emails (Update existing, Insert new)
        existing_emails = set(Customer.objects.values_list('Email', flat=True))  # Get all existing emails
        
        customers_to_create = []
        customers_to_update = []

        for _, row in df.iterrows():
            customer_data = {
                'customerID': row.get('customerID', 'N/A'),
                'gender': row.get('gender', 0),
                'SeniorCitizen': row.get('SeniorCitizen', 0),
                'Partner': row.get('Partner', 0),
                'Dependents': row.get('Dependents', 0),
                'tenure': row.get('tenure', 0),
                'PhoneService': row.get('PhoneService', 0),
                'MultipleLines': row.get('MultipleLines', 0),
                'InternetService': row.get('InternetService', "Unknown"),
                'OnlineSecurity': row.get('OnlineSecurity', 0),
                'OnlineBackup': row.get('OnlineBackup', 0),
                'DeviceProtection': row.get('DeviceProtection', 0),
                'TechSupport': row.get('TechSupport', 0),
                'StreamingTV': row.get('StreamingTV', 0),
                'StreamingMovies': row.get('StreamingMovies', 0),
                'Contract': row.get('Contract', "Unknown"),
                'PaperlessBilling': row.get('PaperlessBilling', 0),
                'PaymentMethod': row.get('PaymentMethod', "Unknown"),
                'MonthlyCharges': row.get('MonthlyCharges', 0.0),
                'TotalCharges': row.get('TotalCharges', 0.0),
                'Churn': row.get('Churn_Prediction', 0),
                'Age': row.get('Age', 0),
                'SatisfactionScore': row.get('SatisfactionScore', 0),
                'CustomerSupportCalls': row.get('CustomerSupportCalls', 0),
                'PaymentTimeliness': row.get('PaymentTimeliness', 0),
                'LifetimeValue': row.get('LifetimeValue', 0.0),
                'AverageDailyUsage': row.get('AverageDailyUsage', 0.0),
                'Email': row.get('Email', "unknown@example.com"),
            }

            if customer_data['Email'] in existing_emails:
                # Update existing record
                existing_customer = Customer.objects.get(Email=customer_data['Email'])
                for key, value in customer_data.items():
                    setattr(existing_customer, key, value)
                customers_to_update.append(existing_customer)
            else:
                # Add new record
                customers_to_create.append(Customer(**customer_data))

        # Perform bulk updates and inserts
        if customers_to_update:
            Customer.objects.bulk_update(customers_to_update, [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges', 'Churn', 'Age', 'SatisfactionScore',
                'CustomerSupportCalls', 'PaymentTimeliness', 'LifetimeValue', 'AverageDailyUsage'
            ])

        if customers_to_create:
            Customer.objects.bulk_create(customers_to_create)

        return JsonResponse({'message': f'{len(customers_to_create)} new records added, {len(customers_to_update)} records updated!'}, status=200)

    return JsonResponse({'error': 'No file uploaded'}, status=400)