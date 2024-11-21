from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model

User = get_user_model()

class EmailBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        """Allow users to authenticate with email instead of username."""
        # Check if username is provided as an email
        try:
            # Fetch the user by email instead of username
            user = User.objects.get(email=username)
        except User.DoesNotExist:
            return None
        
        # Check the password if the user exists
        if user.check_password(password):
            return user
        return None

    def get_user(self, user_id):
        """Retrieve a user by ID."""
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
