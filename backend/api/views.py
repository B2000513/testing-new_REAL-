from django.shortcuts import render

# Create your views here.
from api.models import User, Profile
from api.serializers import UserSerializer, MyTokenObtainPairSerializer, RegisterSerializer ,ProfileSerializer
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
        '/api/profile/update/'
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