import pandas as pd
import openai # Install with `pip install openai`
import os
from dotenv import load_dotenv
from django.shortcuts import render , redirect

# Create your views here.
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
        '/api/token/refresh/'
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
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])  # Protect API access
def upload_customers_from_excel(request):
    if 'excel_file' not in request.FILES:
        return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

    excel_file = request.FILES['excel_file']
    fs = FileSystemStorage()
    filename = fs.save(excel_file.name, excel_file)
    uploaded_file_path = fs.path(filename)

    try:
        df = pd.read_excel(uploaded_file_path)

        
        with transaction.atomic():  # Ensure atomicity
            # **1. Clear all existing customer records**
            Customer.objects.all().delete()

            # **2. Store unique emails to avoid duplication in the new upload**
            unique_emails = set()

            customers = []
            for _, row in df.iterrows():
                email = str(row.get('Email', '')).strip()

                # Skip empty or duplicate emails
                if not email or email in unique_emails:
                    continue  
                
                unique_emails.add(email)  # Add email to the set

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
                    Churn=row.get('Churn', 0),
                    Age=row.get('Age', 0),
                    SatisfactionScore=row.get('SatisfactionScore', 0),
                    CustomerSupportCalls=row.get('CustomerSupportCalls', 0),
                    PaymentTimeliness=row.get('PaymentTimeliness', 0),
                    LifetimeValue=row.get('LifetimeValue', 0.0),
                    AverageDailyUsage=row.get('AverageDailyUsage', 0.0),
                    Email=email
                ))

            # **3. Bulk insert for efficiency**
            Customer.objects.bulk_create(customers)

        return Response({"message": "Data imported successfully!"}, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response({"error": f"Error occurred: {e}"}, status=status.HTTP_400_BAD_REQUEST)


    finally:
        if fs.exists(filename):  # Ensure file exists before deleting
            fs.delete(filename)

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
    

# AI Chatbot function
def chatbot(request):
    user_message = request.GET.get("message", "")
    if not user_message:
        return JsonResponse({"response": "Please provide a message."})

    # Call AI model (OpenAI API)
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your API key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful customer support chatbot."},
                  {"role": "user", "content": user_message}]
    )
    
    bot_response = response["choices"][0]["message"]["content"]
    return JsonResponse({"response": bot_response})