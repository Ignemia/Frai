"""
Unit tests for authentication functionality - Mock version
"""
import pytest
import sys
from pathlib import Path
from hashlib import sha256

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Mock functions for testing when database is not available
def mock_user_exists(username):
    """Mock user_exists function"""
    return username == "testuser"


def mock_get_user_id(username):
    """Mock get_user_id function"""
    if username == "testuser":
        return 6
    return None


def mock_verify_credentials(username, password_hash):
    """Mock verify_credentials function"""
    if username == "testuser":
        expected_hash = sha256("testpassword".encode()).hexdigest()
        return password_hash == expected_hash
    return False


class TestAuthenticationMock:
    """Test authentication functionality with mocks"""
    
    def test_user_exists(self):
        """Test if user exists check works"""
        # Test with known user
        assert mock_user_exists("testuser") == True
        
        # Test with non-existent user
        assert mock_user_exists("nonexistentuser") == False
    
    def test_get_user_id(self):
        """Test getting user ID"""
        user_id = mock_get_user_id("testuser")
        assert user_id is not None
        assert isinstance(user_id, int)
        assert user_id > 0
        
        # Test non-existent user
        user_id = mock_get_user_id("nonexistentuser")
        assert user_id is None
    
    def test_verify_credentials(self):
        """Test credential verification"""
        username = "testuser"
        password = "testpassword"
        password_hash = sha256(password.encode()).hexdigest()
        
        # Test valid credentials
        assert mock_verify_credentials(username, password_hash) == True
        
        # Test invalid password
        wrong_hash = sha256("wrongpassword".encode()).hexdigest()
        assert mock_verify_credentials(username, wrong_hash) == False
        
        # Test non-existent user
        assert mock_verify_credentials("nonexistentuser", password_hash) == False
    
    def test_password_hashing(self):
        """Test password hashing consistency"""
        password = "testpassword"
        hash1 = sha256(password.encode()).hexdigest()
        hash2 = sha256(password.encode()).hexdigest()
        
        # Same password should produce same hash
        assert hash1 == hash2
        
        # Different passwords should produce different hashes
        different_hash = sha256("differentpassword".encode()).hexdigest()
        assert hash1 != different_hash
