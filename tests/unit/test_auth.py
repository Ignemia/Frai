"""
Unit tests for authentication functionality
"""
import pytest
import sys
from pathlib import Path
from hashlib import sha256

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.database.users import user_exists, get_user_id
from services.database.passwords import verify_credentials, get_password_hash


class TestAuthentication:
    """Test authentication functionality"""
    
    def test_user_exists(self):
        """Test if user exists check works"""
        # Test with known user
        assert user_exists("testuser") == True
        
        # Test with non-existent user
        assert user_exists("nonexistentuser") == False
    
    def test_get_user_id(self):
        """Test getting user ID"""
        user_id = get_user_id("testuser")
        assert user_id is not None
        assert isinstance(user_id, int)
        assert user_id > 0
    
    def test_verify_credentials(self):
        """Test credential verification"""
        username = "testuser"
        password = "testpassword"
        password_hash = sha256(password.encode()).hexdigest()
        
        # Test valid credentials
        assert verify_credentials(username, password_hash) == True
        
        # Test invalid password
        wrong_hash = sha256("wrongpassword".encode()).hexdigest()
        assert verify_credentials(username, wrong_hash) == False
        
        # Test non-existent user
        assert verify_credentials("nonexistentuser", password_hash) == False
